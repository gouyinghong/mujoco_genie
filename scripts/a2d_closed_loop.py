"""A2D closed-loop gripper with one finite-torque drive per hand.

Inspired by the connect/synchronization/tendon structure of Robotiq 2F85 in
hangtingLiu/VLM_Grasp_Interactive. All dimensions and signs here are A2D-specific.
The extra linkage coupling is fitted to A2D's existing opening calibration;
it is an effective transmission model, not a hardware-validated gear model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from weakref import WeakKeyDictionary

import mujoco
import numpy as np

from scripts.replay_a2d import (
    A2D_GRIPPER_JOINTS, A2D_ARM_JOINT_NAMES, A2D_UPPER_BODY_POSE,
    GRIPPER_WIDE_JOINT_POSITIONS, set_gripper_command,
)


def add_gripper_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm-contact-mode", choices=("legacy", "constrained"), default="legacy",
                        help="constrained makes the contact solver respect prescribed arm/torso motion")
    parser.add_argument("--physics-timestep", type=float, default=None,
                        help="Physics step in seconds; independent of viewer playback speed")
    parser.add_argument("--contact-impratio", type=float, default=10.0,
                        help="MuJoCo friction-to-normal constraint impedance ratio")
    parser.add_argument("--gripper-control", choices=("closed-loop", "kinematic"),
                        default="closed-loop")
    parser.add_argument("--gripper-kp", type=float, default=30.0)
    parser.add_argument("--gripper-kv", type=float, default=0.2)
    parser.add_argument("--gripper-max-torque", type=float, default=1.0,
                        help="Drive torque limit per hand, shared by the two fingers (N m)")
    parser.add_argument("--gripper-close-bias", type=float, default=0.15,
                        help="Extra closure target: clip(openness - bias*(1-openness), 0, 1)")
    parser.add_argument("--gripper-release-mode", choices=("recorded", "fast"), default="recorded",
                        help="fast commands full opening when the recorded command starts increasing; torque stays limited")
    parser.add_argument("--gripper-sliding-friction", type=float, default=None,
                        help="Override fingertip sliding friction in closed-loop mode; default keeps model values")
    parser.add_argument("--grasp-lower-m", type=float, default=0.0,
                        help="Optional downward translation of the right arm trajectory in memory; source data is unchanged")


def lower_grasp_trajectory(model, trajectory, bindings, upper_body_pose, distance_m):
    """Optional IK correction on a copy, using the same solver as layout preparation."""
    from scripts.a2d_batch import translate_right_arm_trajectory

    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("grasp-lower-m must be finite and nonnegative")
    corrected, metrics = translate_right_arm_trajectory(
        model, trajectory, bindings, upper_body_pose, -distance_m
    )
    if metrics["max_position_error_m"] > 0.001:
        raise ValueError(f"Grasp height correction did not converge: {metrics}")
    return corrected


def load_physics_model(path: str | Path, *, gripper_control: str = "closed-loop",
                       gripper_kp: float = 30.0, gripper_kv: float = 0.2,
                       gripper_max_torque: float = 1.0,
                       gripper_sliding_friction: float | None = None,
                       arm_contact_mode: str = "legacy",
                       physics_timestep: float | None = None,
                       contact_impratio: float = 10.0) -> mujoco.MjModel:
    values = (gripper_kp, gripper_kv, gripper_max_torque)
    if not np.all(np.isfinite(values)) or min(gripper_kp, gripper_max_torque) <= 0 or gripper_kv < 0:
        raise ValueError("Gripper kp/torque must be positive and kv nonnegative, all finite")
    if gripper_sliding_friction is not None and (
        not np.isfinite(gripper_sliding_friction) or gripper_sliding_friction < 0
    ):
        raise ValueError("Gripper sliding friction must be finite and nonnegative")
    if arm_contact_mode not in ("legacy", "constrained"):
        raise ValueError("Unknown arm contact mode")
    if not np.isfinite(contact_impratio) or contact_impratio <= 0:
        raise ValueError("Contact impratio must be finite and positive")
    if physics_timestep is not None and (not np.isfinite(physics_timestep) or physics_timestep <= 0):
        raise ValueError("Physics timestep must be finite and positive")
    if gripper_control == "kinematic":
        if arm_contact_mode != "legacy" or physics_timestep is not None or contact_impratio != 10:
            raise ValueError("Solver overrides require closed-loop gripper mode")
        if gripper_sliding_friction is not None:
            raise ValueError("Gripper sliding friction override requires closed-loop mode")
        return mujoco.MjModel.from_xml_path(str(path))
    if gripper_control != "closed-loop":
        raise ValueError("Unknown gripper control mode")
    spec = mujoco.MjSpec.from_file(str(Path(path).resolve()))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = contact_impratio
    if physics_timestep is not None:
        spec.option.timestep = physics_timestep
    if arm_contact_mode == "constrained":
        for name in (*A2D_ARM_JOINT_NAMES, *(name for name, _ in A2D_UPPER_BODY_POSE)):
            spec.add_equality(
                name=f"prescribed_{name}", type=mujoco.mjtEq.mjEQ_JOINT,
                objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=name,
                data=[0] * 11, solref=[0.004, 1],
                solimp=[0.9999, 0.9999, 0.001, 0.5, 2],
            )
    x, y = GRIPPER_WIDE_JOINT_POSITIONS[:, :2].T
    # Preserve the authored zero configuration exactly. Link 3's nonlinear
    # coupling removes the extra DOF left by the five-pivot loop. Links 2 and 4
    # remain passive and are solved by the geometric closure.
    coefficients = np.r_[0.0, np.linalg.lstsq(
        np.column_stack([x**i for i in range(1, 5)]), y, rcond=None
    )[0]]
    constraint = dict(solref=[0.004, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2])
    for side in ("left", "right"):
        for finger, sign in (("wide", 1), ("narrow", -1)):
            for link in (1, 2, 3, 4):
                joint = spec.joint(f"{side}_{finger}{link}_joint")
                joint.armature = 0.0001
                joint.damping = [0.002, 0, 0]
            # Link 4 has a second pin at y=+/-10.5 mm in its STL. MuJoCo
            # derives the matching link-2 anchor from the authored zero pose.
            spec.add_equality(
                name=f"{side}_{finger}_loop", type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=f"{side}_{finger}4_Link", name2=f"{side}_{finger}2_Link",
                data=[0, sign * 0.0105, 0] + [0] * 8, **constraint,
            )
            signed = coefficients * np.array([sign ** (i + 1) for i in range(5)])
            spec.add_equality(
                name=f"{side}_{finger}_coupling", type=mujoco.mjtEq.mjEQ_JOINT,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                name1=f"{side}_{finger}3_joint", name2=f"{side}_{finger}1_joint",
                data=list(signed) + [0] * 6, **constraint,
            )
        spec.add_equality(
            name=f"{side}_finger_sync", type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            name1=f"{side}_wide1_joint", name2=f"{side}_narrow1_joint",
            data=[0, -1, 0, 0, 0] + [0] * 6, **constraint,
        )
        tendon = spec.add_tendon(name=f"{side}_gripper_drive")
        tendon.wrap_joint(f"{side}_wide1_joint", 0.5)
        tendon.wrap_joint(f"{side}_narrow1_joint", -0.5)
        actuator = spec.add_actuator(
            name=f"{side}_gripper_actuator", target=f"{side}_gripper_drive",
            trntype=mujoco.mjtTrn.mjTRN_TENDON,
            forcelimited=True, forcerange=[-gripper_max_torque, gripper_max_torque],
            ctrllimited=True, ctrlrange=[0, np.pi / 4],
        )
        actuator.set_to_position(kp=gripper_kp, kv=gripper_kv)
    model = spec.compile()
    for side in ("left", "right"):
        for suffix in ("narrow_fingertip_collision", "wide_fingertip_lower_collision",
                       "wide_fingertip_upper_collision"):
            geom = model.geom(f"{side}_{suffix}").id
            model.geom_solref[geom] = (0.004, 1)
            model.geom_priority[geom] = 1
            if gripper_sliding_friction is not None:
                model.geom_friction[geom, 0] = gripper_sliding_friction
    return model


_prescribed_indices = WeakKeyDictionary()


def update_prescribed_arm_constraints(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Expose the already prescribed q/qvel to the contact solve.

    These ideal trajectory supports are for kinematic arm replay, not a finite
    torque arm controller. Finger joints and the free die are never constrained.
    The reference offset cancels constraint damping at the requested velocity:
    b/k = 2 * timeconst * dampratio**2 for our constant-impedance
    equalities (d(r) = d_width, positive solref).
    """
    indices = _prescribed_indices.get(model)
    if indices is None:
        eq = np.array([i for i in range(model.neq)
                       if (model.equality(i).name or "").startswith("prescribed_")], dtype=int)
        joints = model.eq_obj1id[eq]
        indices = eq, model.jnt_qposadr[joints].copy(), model.jnt_dofadr[joints].copy()
        _prescribed_indices[model] = indices
    eq, q, v = indices
    if not eq.size:
        return
    tau = np.maximum(model.eq_solref[eq, 0], 2 * model.opt.timestep)
    offset = 2 * tau * model.eq_solref[eq, 1] ** 2
    model.eq_data[eq, 0] = data.qpos[q] - model.qpos0[q] + offset * data.qvel[v]


class ClosedLoopGripper:
    def __init__(self, model: mujoco.MjModel, close_bias: float = 0.15,
                 release_mode: str = "recorded"):
        if not np.isfinite(close_bias) or not 0 <= close_bias <= 1:
            raise ValueError("gripper-close-bias must be finite and in [0, 1]")
        if release_mode not in ("recorded", "fast"):
            raise ValueError("Unknown gripper release mode")
        self.model, self.close_bias = model, close_bias
        self.release_mode = release_mode
        self.previous_openness = None
        self.releasing = np.zeros(2, dtype=bool)
        self.actuator_ids = np.array([
            model.actuator(f"{side}_gripper_actuator").id for side in ("left", "right")
        ])
        ids = [model.joint(name).id for name in A2D_GRIPPER_JOINTS]
        self.qpos_addresses = model.jnt_qposadr[ids].copy()
        self.dof_addresses = model.jnt_dofadr[ids].copy()

    def command(self, data: mujoco.MjData, openness: np.ndarray, *,
                minimum: float = 0, initialize: bool = False) -> None:
        openness = np.asarray(openness, dtype=float)
        if openness.shape != (2,) or not np.all(np.isfinite(openness)):
            raise ValueError("Expected two finite gripper commands")
        if initialize:
            self.previous_openness = None
            self.releasing[:] = False
        if self.previous_openness is not None:
            # A new closing command cancels fast release independently per hand.
            self.releasing |= (openness > self.previous_openness + 1e-8) & (self.previous_openness < .99)
            self.releasing &= ~(openness < self.previous_openness - 1e-8)
        targets = np.clip(openness - self.close_bias * (1 - openness), minimum, 1)
        if self.release_mode == "fast":
            targets[self.releasing] = 1
        data.ctrl[self.actuator_ids] = targets * np.pi / 4
        self.previous_openness = openness.copy()
        if initialize:
            set_gripper_command(self.model, data, targets)


def loop_error_m(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    errors = []
    for side in ("left", "right"):
        for finger in ("wide", "narrow"):
            eq = model.equality(f"{side}_{finger}_loop").id
            a, b = model.eq_obj1id[eq], model.eq_obj2id[eq]
            anchors = model.eq_data[eq]
            errors.append(np.linalg.norm(
                data.xpos[a] + data.xmat[a].reshape(3, 3) @ anchors[:3]
                - data.xpos[b] - data.xmat[b].reshape(3, 3) @ anchors[3:6]
            ))
    return float(max(errors))
