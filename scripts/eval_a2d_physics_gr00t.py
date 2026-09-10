#!/usr/bin/env python3
"""Evaluate a remote GR00T policy in collected A2D physics test scenes.

Dataset vectors: left arm(7), right arm(7), left/right aperture(2).
Only the initial robot state is read from the demonstration. Subsequent motion
comes exclusively from policy targets, including finite-torque finger dynamics.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_mujoco_gr00t_genie1 as protocol
from scripts.collect_a2d_physics_lerobot import legacy, make_scene, write_json
from scripts.a2d_closed_loop import update_prescribed_arm_constraints, loop_error_m
from scripts.a2d_dice_orientation import dice_quaternion
from scripts.a2d_batch import _is_descendant
from scripts.replay_a2d import set_upper_body_pose
from scripts.replay_a2d_physics import set_robot_target, set_initial_dice_pose

DEFAULT_DATASET = Path('datasets/a2d_augmented_v2/collection/test/lerobot')


def create_policy_client(args):
    class Client(protocol.import_policy_client()):
        def _init_socket(self):
            # The reference client replaces a timed-out socket without closing it.
            # Close it first so context termination cannot hang after a timeout.
            old = getattr(self, 'socket', None)
            if old is not None:
                old.close(linger=0)
            super()._init_socket()

    return Client(host=args.policy_host, port=args.policy_port,
                  timeout_ms=args.policy_timeout_ms, strict=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--episode-index', type=int, default=0, help='LeRobot test episode index, not source filename suffix')
    p.add_argument('--all-episodes', action='store_true')
    p.add_argument('--policy-host', default=protocol.DEFAULT_POLICY_HOST)
    p.add_argument('--policy-port', type=int, default=protocol.DEFAULT_POLICY_PORT)
    p.add_argument('--policy-timeout-ms', type=int, default=30000)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--replan-steps', type=int, default=8, help='Execute at most N actions per response; 0 uses full chunk')
    p.add_argument('--flat-action-order', choices=('arms-first','interleaved'), default='arms-first')
    p.add_argument('--max-joint-speed', type=float, default=2., help='rad/s arm target limit; interventions logged')
    p.add_argument('--success-hold-steps', type=int, default=10)
    p.add_argument('--language', help='Default: task from collection report')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--preview', action='store_true', help='Show policy camera input in a separate window')
    p.add_argument('--start-paused', action='store_true', help='GUI starts paused; SPACE resumes')
    p.add_argument('--realtime', action='store_true', help='Pace control steps to wall clock; inference still pauses physics')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate each selected initial scene/image, without contacting server')
    p.add_argument('--output-dir', type=Path, default=Path('logs')/('a2d_gr00t_physics_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    args=p.parse_args(argv)
    if args.max_steps<=0 or args.replan_steps<0 or args.policy_timeout_ms<=0 or args.success_hold_steps<=0:
        p.error('steps/timeout must be positive; replan-steps may be zero')
    if not np.isfinite(args.max_joint_speed) or args.max_joint_speed<=0:
        p.error('max-joint-speed must be finite and positive')
    if args.headless and (args.preview or args.start_paused):
        p.error('headless cannot be combined with preview/start-paused')
    return args


def load_scenes(dataset):
    dataset=dataset.expanduser().resolve()
    info=json.loads((dataset/'meta/info.json').read_text())
    report=json.loads((dataset.parent/'collection_report.json').read_text())
    if report['status']!='complete':
        raise ValueError('Collection is incomplete')
    if info['fps']!=report['parameters']['fps'] or not 0<info['fps']<=2000:
        raise ValueError('Inconsistent collection frame rate')
    names=info['features']['observation.state']['names']
    if len(names)!=16 or names[-2:]!=list(legacy.GRIPPER_NAMES) or info['features']['action']['names']!=names:
        raise ValueError('Expected arms-first 14 joint + 2 aperture dataset schema')
    expected=dict(timestep_s=.0005,arm_contact_mode='constrained',impratio=100,gripper_kp=30,
                  gripper_kv=.2,max_torque_nm=1,friction=3,close_bias=0,grasp_lower_m=0,
                  settle_time_s=.4,dice_linear_damping=.02,dice_angular_damping=.0005)
    if report['physics']!=expected:
        raise ValueError('Collection physics differs from supported scene loader; do not silently evaluate with other physics')
    return dataset,info,report


def action_chunk(response, flat_order='arms-first'):
    if 'action' in response:
        array=protocol.normalize_action_array(response['action'])
        if array.shape[0]!=1 or array.shape[-1]!=16 or array.shape[1]==0:
            raise ValueError(f'Invalid flat action chunk: {array.shape}')
        rows=array[0]
        if flat_order=='interleaved':
            rows=rows[:,[*range(7),*range(8,15),7,15]]
    else:
        rows=np.array(list(protocol.iter_action_chunk(response)))
        if rows.size==0:
            raise ValueError('Empty structured action chunk')
        rows=rows[:,[*range(7),*range(8,15),7,15]]
    if not np.isfinite(rows).all():
        raise ValueError('Policy returned NaN/Inf')
    return [r.astype(np.float64,copy=True) for r in rows]


def observation(image,state,language):
    # Wire protocol is named modalities; dataset storage remains arms-first.
    interleaved=protocol.compose_policy_state(state[:14],state[14:])
    return protocol.make_policy_observation(image,interleaved,language)


class PhysicsEnv:
    def __init__(self, entry, info, report):
        path=Path(entry['prepared_manifest'])
        self.model,tr,self.bindings,self.pose,_,self.record,self.gripper=make_scene(path,'recorded')
        if list(tr.joint_names)!=info['features']['observation.state']['names'][:14]:
            raise ValueError('Scene joints and dataset feature order disagree')
        self.fps=info['fps'];self.dt=float(self.model.opt.timestep);self.steps=0;self.ticks=0
        self.data=mujoco.MjData(self.model);self.visual=mujoco.MjData(self.model)
        self.target=np.r_[tr.joint_positions[0],tr.effector_positions[0]].astype(float)
        set_robot_target(self.model,self.data,tr,self.bindings,0.,self.pose,0.,moving=False,
                         gripper=self.gripper,initialize_gripper=True)
        d=self.record['dice']
        set_initial_dice_pose(self.model,self.data,np.array(d['initial_position']),dice_quaternion(d))
        mujoco.mj_forward(self.model,self.data)
        for _ in range(round(.4/self.dt)):
            self._target(self.target[:14],np.zeros(14),self.target[14:])
            mujoco.mj_step(self.model,self.data)
        self.initial_height=float(d['initial_position'][2])
        self.peak_height=self.initial_height;self.grasp_ticks=0;self.grasped=False;self.lifted=False
        self.max_penetration=0.;self.max_loop_error=0.;self.max_torque=0.
        self.dice_id=self.model.body('dice').id;self.box_id=self.model.body('cardboard_box').id
        self.dice_geom=self.model.geom('dice_collision').id
        self.dice_dof=self.model.joint('dice_free_joint').dofadr[0]
        self.dice_qadr=self.model.joint('dice_free_joint').qposadr[0]
        self.finger_groups={}
        for side in ('left','right'):
            self.finger_groups[side]=({self.model.geom(f'{side}_narrow_fingertip_collision').id},
                                    {self.model.geom(f'{side}_wide_fingertip_{i}_collision').id for i in ('lower','upper')})
        self.robot_bodies={i for i in range(self.model.nbody) if _is_descendant(self.model,i,self.model.body('base_link').id)}
        self.support_bodies={self.model.body('table').id,self.box_id}
        self.aperture_addresses=[(self.model.joint(f'{s}_wide1_joint').qposadr[0],self.model.joint(f'{s}_narrow1_joint').qposadr[0]) for s in ('left','right')]
        self.refresh()
        self.report=report;self.entry=entry

    def _target(self,q,v,grip):
        self.data.qpos[self.bindings.qpos_addresses]=q
        self.data.qvel[self.bindings.dof_addresses]=v
        set_upper_body_pose(self.model,self.data,self.pose)
        update_prescribed_arm_constraints(self.model,self.data)
        # Targets already represent effective actuator aperture: no extra fast-opening latch.
        self.data.ctrl[self.gripper.actuator_ids]=grip*np.pi/4

    def refresh(self):
        mujoco.mj_copyData(self.visual,self.model,self.data)
        mujoco.mj_forward(self.model,self.visual)

    def state(self):
        aperture=np.array([.5*(self.data.qpos[a]-self.data.qpos[b])/(np.pi/4) for a,b in self.aperture_addresses])
        return legacy.compose_robot_state(self.data.qpos[self.bindings.qpos_addresses],np.clip(aperture,0,1))

    def contacts(self):
        loaded=set();force=np.zeros(6)
        for i,c in enumerate(self.data.contact):
            g1,g2=int(c.geom1),int(c.geom2)
            b1,b2=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
            if (b1 in self.robot_bodies and b2 in self.support_bodies) or (b2 in self.robot_bodies and b1 in self.support_bodies):
                self.max_penetration=max(self.max_penetration,-float(c.dist))
            if self.dice_geom in (g1,g2):
                mujoco.mj_contactForce(self.model,self.data,i,force)
                if force[0]>.001:loaded.add(g2 if g1==self.dice_geom else g1)
        bilateral=any(bool(loaded&a) and bool(loaded&b) for a,b in self.finger_groups.values())
        any_finger=any(bool(loaded&(a|b)) for a,b in self.finger_groups.values())
        return bilateral,any_finger

    def advance(self,proposed,max_speed):
        proposed=np.asarray(proposed,dtype=float)
        if proposed.shape!=(16,) or not np.isfinite(proposed).all():raise ValueError('Expected finite 16-D policy target')
        bounded=proposed.copy()
        for i,jid in enumerate(self.bindings.joint_ids):
            if self.model.jnt_limited[jid]:bounded[i]=np.clip(bounded[i],*self.model.jnt_range[jid])
        bounded[14:]=np.clip(bounded[14:],0,1)
        end_tick=round((self.steps+1)/self.fps/self.dt);n=end_tick-self.ticks;duration=n*self.dt
        bounded[:14]=np.clip(bounded[:14],self.target[:14]-max_speed*duration,self.target[:14]+max_speed*duration)
        clipped=not np.allclose(bounded,proposed,atol=1e-8,rtol=0)
        start=self.target.copy();velocity=(bounded[:14]-start[:14])/duration
        for j in range(n):
            self._target(start[:14]+j/n*(bounded[:14]-start[:14]),velocity,bounded[14:])
            mujoco.mj_step(self.model,self.data)
            bilateral,_=self.contacts()
            self.grasp_ticks=self.grasp_ticks+1 if bilateral else 0
            self.grasped |= self.grasp_ticks*self.dt>=.1
            height=float(self.data.xpos[self.dice_id,2]);self.peak_height=max(self.peak_height,height)
            self.lifted |= self.grasped and bilateral and height-self.initial_height>=.08
            self.max_loop_error=max(self.max_loop_error,loop_error_m(self.model,self.data))
            self.max_torque=max(self.max_torque,float(np.max(np.abs(self.data.actuator_force[self.gripper.actuator_ids]))))
        self.target=bounded;self.steps+=1;self.ticks=end_tick;self.refresh()
        return bounded.copy(),clipped

    def metrics(self):
        v=self.visual;rotation=v.xmat[self.box_id].reshape(3,3)
        local=rotation.T@(v.xpos[self.dice_id]-v.xpos[self.box_id])
        extent=np.abs(rotation.T@v.xmat[self.dice_id].reshape(3,3))@np.full(3,.03)
        inner=np.array([self.model.geom(f'cardboard_box_collision_wall_{axis}_positive').pos[i]-self.model.geom(f'cardboard_box_collision_wall_{axis}_positive').size[i] for i,axis in enumerate(('x','y'))])
        landed=bool(np.all(np.abs(local[:2])+extent[:2]<=inner+.001) and -.001<=local[2]-extent[2]<=.01 and np.linalg.norm(self.data.qvel[self.dice_dof:self.dice_dof+3])<.01)
        _,contact=self.contacts();warnings=int(sum(w.number for w in self.data.warning))
        good=bool(self.grasped and self.lifted and landed and not contact and warnings==0 and self.max_loop_error<.001 and self.max_penetration<=.001)
        return dict(grasped=bool(self.grasped),lifted=bool(self.lifted),landed_in_box=landed,
                    finger_contact=contact,peak_height_m=self.peak_height,physics_warnings=warnings,
                    max_loop_error_m=self.max_loop_error,max_robot_support_penetration_m=self.max_penetration,
                    max_drive_torque_nm=self.max_torque,placement_condition=good)

    def renderer(self):
        m=self.model;m.vis.global_.offwidth=legacy.HEAD_CAMERA_WIDTH;m.vis.global_.offheight=legacy.HEAD_CAMERA_HEIGHT
        legacy.hide_closed_head_shell(m);legacy.set_target_visibility(m,False)
        appearance=self.entry.get('augmentation',{}).get('appearance',{})
        legacy.apply_texture_gamma(m,'cardboard_box_texture',appearance.get('box_texture_gamma',.65))
        if appearance:
            m.light_diffuse[:]*=appearance['light_scale'];m.geom_rgba[m.geom('table_top').id,:3]=appearance['table_rgb']
        option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[legacy.CAMERA_OCCLUDER_GROUP]=0
        maps=None if self.report['parameters']['no_distortion'] else legacy.distortion_maps()
        renderer=mujoco.Renderer(m,height=legacy.HEAD_CAMERA_HEIGHT,width=legacy.HEAD_CAMERA_WIDTH)
        return renderer,option,maps


def run_episode(args,entry,info,report,output,policy):
    renderer=None;viewer=None;writer=None
    traces={'state':[],'proposed_action':[],'action':[],'state_time_s':[],'next_time_s':[],'next_state':[],'next_dice_qpos':[]}
    result={'status':'running','lerobot_episode_index':entry['lerobot_episode_index'],
            'source_episode':entry['source_episode'],'prepared_manifest':entry['prepared_manifest'],
            'augmentation':entry.get('augmentation'),'success':False,'steps':0,'clipped_actions':0,'inference_calls':0}
    paused=[args.start_paused];streak=0
    def key(k):
        if k==32:paused[0]=not paused[0]
    try:
        env=PhysicsEnv(entry,info,report)
        renderer,option,maps=env.renderer()
        image=protocol.render_policy_image(renderer,env.visual,option,maps).copy()
        cv2.imwrite(str(output/'initial.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if args.dry_run:
            observation(image,env.state(),args.language)
            result.update(status='dry_run',metrics=env.metrics())
            return result
        policy.reset({'language':args.language})
        if not args.headless:
            from mujoco import viewer as mv
            viewer=mv.launch_passive(env.model,env.data,key_callback=key,show_left_ui=False,show_right_ui=False)
            with viewer.lock():
                # Match replay's visual mesh view; group 0 contains collision proxies.
                viewer.opt.geomgroup[0] = 0
                viewer.opt.geomgroup[1] = 1
                viewer.opt.geomgroup[2] = 1
        if args.preview:cv2.namedWindow('GR00T physics input',cv2.WINDOW_NORMAL)
        if args.save_video:writer=protocol.VideoWriter(output/'rollout.mp4',env.fps)
        queue=deque()
        for step in range(args.max_steps):
            while paused[0]:
                if viewer is not None and not viewer.is_running():result['status']='aborted';return result
                if viewer is not None:viewer.sync()
                time.sleep(.01)
            if viewer is not None and not viewer.is_running():result['status']='aborted';break
            start=time.perf_counter()
            env.refresh();image=protocol.render_policy_image(renderer,env.visual,option,maps).copy();state=env.state()
            if args.preview:
                cv2.imshow('GR00T physics input',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1)&0xff in (27,ord('q')):result['status']='aborted';break
            if not queue:
                response,_=policy.get_action(observation(image,state,args.language))
                chunk=action_chunk(response,args.flat_action_order)
                queue.extend(chunk[:args.replan_steps] if args.replan_steps else chunk)
                result['inference_calls']+=1
                print(f'  inference step={step} chunk={len(queue)}',flush=True)
            proposed=queue.popleft()
            applied,clipped=env.advance(proposed,args.max_joint_speed)
            if writer:writer.append(image)
            traces['state'].append(state);traces['proposed_action'].append(proposed);traces['action'].append(applied)
            traces['state_time_s'].append(round((env.steps-1)/env.fps/env.dt)*env.dt)
            traces['next_time_s'].append(env.ticks*env.dt)
            traces['next_state'].append(env.state())
            traces['next_dice_qpos'].append(env.data.qpos[env.dice_qadr:env.dice_qadr+7].copy())
            result['steps']=env.steps;result['clipped_actions']+=int(clipped)
            metrics=env.metrics();result['metrics']=metrics
            streak=streak+1 if metrics['placement_condition'] else 0
            if viewer is not None:viewer.sync()
            if streak>=args.success_hold_steps:result.update(status='success',success=True);break
            if metrics['physics_warnings'] or not np.isfinite(env.data.qpos).all():
                result['status']='unstable';break
            if args.realtime:time.sleep(max(0.,1/env.fps-(time.perf_counter()-start)))
        if result['status']=='running':result['status']='timeout'
    except BaseException as exc:
        result.update(status='aborted' if isinstance(exc,KeyboardInterrupt) else 'error',
                      error=f'{type(exc).__name__}: {exc}')
        if isinstance(exc,KeyboardInterrupt):raise
    finally:
        if writer is not None:writer.close()
        if viewer is not None:viewer.close()
        if renderer is not None:renderer.close()
        if args.preview:cv2.destroyAllWindows()
        np.savez_compressed(output/'rollout.npz',**{k:np.asarray(v) for k,v in traces.items()})
        write_json(output/'result.json',result)
    return result


def save_summary(path, summary):
    """Count recorded evaluations; scene-only dry runs have no success rate."""
    episodes = summary['episodes']
    evaluated = [r for r in episodes if r['status'] not in ('dry_run', 'running')]
    total = len(evaluated)
    successes = sum(r.get('success') is True for r in evaluated)
    summary.update(
        evaluated_episodes=total,
        successful_episodes=successes,
        unsuccessful_episodes=total-successes,
        success_rate=successes/total if total else None,
        success_rate_percent=round(100*successes/total, 2) if total else None,
        status_counts=dict(Counter(r['status'] for r in episodes)),
    )
    write_json(path, summary)


def main(argv=None):
    args=parse_args(argv);dataset,info,report=load_scenes(args.dataset)
    entries=report['episodes'] if args.all_episodes else [e for e in report['episodes'] if e['lerobot_episode_index']==args.episode_index]
    if not entries:raise ValueError('No selected test episode')
    args.language=args.language or report['parameters']['task']
    output=args.output_dir.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    summary={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             'dataset':str(dataset),'physics':report['physics'],'episodes':[],
             'selected_episodes':len(entries),
             'action_order':'left_arm(7),right_arm(7),left_gripper,right_gripper',
             'control':'Interpolated arm position targets with prescribed constraints; direct finite-torque gripper actuator targets; physics paused during network inference'}
    policy=None
    try:
        if not args.dry_run:
            policy=create_policy_client(args)
        for entry in entries:
            index=entry['lerobot_episode_index'];sub=output/f'episode_{index:06d}';sub.mkdir()
            print(f'Evaluating test index={index} scene={entry["source_episode"]}',flush=True)
            r=run_episode(args,entry,info,report,sub,policy);summary['episodes'].append(r)
            save_summary(output/'summary.json',summary)
            print(f'  status={r["status"]} success={r["success"]}',flush=True)
            if r['status'] in ('error','aborted'):break
    finally:
        if policy is not None:policy.close()
        save_summary(output/'summary.json',summary)
    if summary['success_rate'] is not None:
        print(f'Success rate: {summary["successful_episodes"]}/{summary["evaluated_episodes"]} '
              f'({summary["success_rate_percent"]:.2f}%)', flush=True)
    print(f'Report: {output/"summary.json"}',flush=True)
    return 0 if len(summary['episodes'])==len(entries) and all(r['status'] in ('success','dry_run') for r in summary['episodes']) else 2


if __name__=='__main__':
    raise SystemExit(main())
