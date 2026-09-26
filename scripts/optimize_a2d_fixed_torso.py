#!/usr/bin/env python3
"""Search a fixed A2D torso pose that maximizes replayable dataset episodes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (  # noqa: E402
    DEFAULT_BODY_LIFT_M,
    DEFAULT_BODY_PITCH_RAD,
    DEFAULT_REPLAY_ROOT,
    DEFAULT_RETARGET_DATASET,
    _right_gripper_center_at_frame,
    choose_dice_center_frame,
    find_collision_free_box_pose,
    fixed_upper_body_pose,
    load_gripper_boundaries,
    load_layout_overrides,
    placement_lift_profile,
    robot_table_metrics,
    translate_right_arm_trajectory,
)
from scripts.convert_a2d_to_mjcf import DEFAULT_A2D_WITH_BOX_MJCF  # noqa: E402
from scripts.replay_a2d import (  # noqa: E402
    JointBindings,
    Trajectory,
    apply_kinematic_pose,
    bind_joints,
    load_trajectory,
)


DEFAULT_DATASET = DEFAULT_RETARGET_DATASET
REPORT_SCHEMA = "a2d_fixed_torso_optimization.v1"
IK_TOLERANCE_M = 0.002
PLACEMENT_LIFTS_M = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)


@dataclass(frozen=True)
class EpisodeInput:
    path: Path
    trajectory: Trajectory
    bindings: JointBindings
    boundaries: dict[str, int]
    center_frame: int
    dice_xy_offset_m: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_A2D_WITH_BOX_MJCF)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPLAY_ROOT / "torso_pose_optimization.json",
    )
    parser.add_argument("--baseline-body-lift-m", type=float, default=DEFAULT_BODY_LIFT_M)
    parser.add_argument(
        "--baseline-body-pitch-rad", type=float, default=DEFAULT_BODY_PITCH_RAD
    )
    parser.add_argument("--pitch-min-rad", type=float, default=0.25)
    parser.add_argument("--pitch-max-rad", type=float, default=0.55)
    parser.add_argument("--pitch-step-rad", type=float, default=0.025)
    parser.add_argument("--height-offset-min-m", type=float, default=0.0)
    parser.add_argument("--height-offset-max-m", type=float, default=0.02)
    parser.add_argument("--height-offset-step-m", type=float, default=0.005)
    parser.add_argument(
        "--reference-height-tolerance-m",
        type=float,
        default=2e-5,
        help="Bisection tolerance when preserving the calibrated grasp height",
    )
    parser.add_argument(
        "--verify-top",
        type=int,
        default=2,
        help="Run full dice/box validation for this many best coarse candidates",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="Skip the slower full dice/box verification stage",
    )
    return parser.parse_args()


def solve_body_lift_for_reference_height(
    model: mujoco.MjModel,
    reference: EpisodeInput,
    body_pitch_rad: float,
    target_z_m: float,
    tolerance_m: float,
) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift_body"
    )
    lower, upper = (float(value) for value in model.jnt_range[joint_id])

    def height(lift_m: float) -> float:
        pose = fixed_upper_body_pose(lift_m, body_pitch_rad)
        return float(
            _right_gripper_center_at_frame(
                model,
                reference.trajectory,
                reference.bindings,
                pose,
                reference.center_frame,
            )[2]
        )

    lower_height = height(lower)
    upper_height = height(upper)
    if not min(lower_height, upper_height) <= target_z_m <= max(
        lower_height, upper_height
    ):
        raise ValueError("Reference grasp height is outside the body-lift range")
    for _ in range(40):
        midpoint = 0.5 * (lower + upper)
        midpoint_height = height(midpoint)
        if abs(midpoint_height - target_z_m) <= tolerance_m:
            return midpoint, midpoint_height
        if (midpoint_height < target_z_m) == (lower_height < upper_height):
            lower = midpoint
        else:
            upper = midpoint
    midpoint = 0.5 * (lower + upper)
    return midpoint, height(midpoint)


def load_episode_inputs(
    model: mujoco.MjModel,
    dataset_dir: Path,
    max_episodes: int | None,
) -> list[EpisodeInput]:
    summary_path = dataset_dir / "retarget_summary.json"
    paths = sorted(dataset_dir.glob("episode_*.npz"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {dataset_dir}")
    overrides = load_layout_overrides(dataset_dir)
    result = []
    for path in paths:
        trajectory = load_trajectory(path, summary_path)
        bindings = bind_joints(model, trajectory.joint_names)
        boundaries = load_gripper_boundaries(path)
        default_center = choose_dice_center_frame(boundaries, trajectory.frames)
        override = overrides.get(path.name, {})
        center_frame = int(override.get("dice_center_frame", default_center))
        dice_offset = tuple(
            float(value)
            for value in override.get("dice_xy_offset_m", (0.0, 0.0))
        )
        result.append(
            EpisodeInput(
                path=path,
                trajectory=trajectory,
                bindings=bindings,
                boundaries=boundaries,
                center_frame=center_frame,
                dice_xy_offset_m=dice_offset,
            )
        )
    return result


def _correct_episode(
    model: mujoco.MjModel,
    episode: EpisodeInput,
    upper_body_pose: tuple[tuple[str, float], ...],
    reference_grasp_z_m: float,
    placement_lift_m: float,
) -> tuple[Trajectory, dict[str, float], float]:
    grasp_z = float(
        _right_gripper_center_at_frame(
            model,
            episode.trajectory,
            episode.bindings,
            upper_body_pose,
            episode.center_frame,
        )[2]
    )
    z_offset_m = reference_grasp_z_m - grasp_z
    offsets = z_offset_m + placement_lift_profile(
        episode.trajectory.frames,
        episode.boundaries,
        placement_lift_m,
    )
    corrected, ik_metrics = translate_right_arm_trajectory(
        model,
        episode.trajectory,
        episode.bindings,
        upper_body_pose,
        offsets,
    )
    return corrected, ik_metrics, z_offset_m


def coarse_candidate_score(
    model: mujoco.MjModel,
    episodes: list[EpisodeInput],
    body_lift_m: float,
    body_pitch_rad: float,
    reference_grasp_z_m: float,
) -> dict[str, Any]:
    pose = fixed_upper_body_pose(body_lift_m, body_pitch_rad)
    records = []
    for episode in episodes:
        try:
            corrected, ik_metrics, z_offset_m = _correct_episode(
                model,
                episode,
                pose,
                reference_grasp_z_m,
                PLACEMENT_LIFTS_M[-1],
            )
            table = robot_table_metrics(model, corrected, episode.bindings, pose)
            status = (
                "ok"
                if ik_metrics["max_position_error_m"] <= IK_TOLERANCE_M
                and table["contact_frames"] == 0
                else "failed"
            )
            records.append(
                {
                    "episode": episode.path.name,
                    "status": status,
                    "right_arm_z_offset_m": z_offset_m,
                    "ik": ik_metrics,
                    "table": table,
                }
            )
        except Exception as error:
            records.append(
                {
                    "episode": episode.path.name,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "body_lift_m": body_lift_m,
        "body_pitch_rad": body_pitch_rad,
        "estimated_usable": sum(record["status"] == "ok" for record in records),
        "episodes": records,
    }


def verified_candidate_score(
    model: mujoco.MjModel,
    episodes: list[EpisodeInput],
    body_lift_m: float,
    body_pitch_rad: float,
    reference_grasp_z_m: float,
) -> dict[str, Any]:
    pose = fixed_upper_body_pose(body_lift_m, body_pitch_rad)
    records = []
    for episode in episodes:
        print(f"    verify {episode.path.name}", flush=True)
        record: dict[str, Any] = {"episode": episode.path.name, "status": "failed"}
        try:
            selected = None
            last_table_safe = None
            for placement_lift_m in PLACEMENT_LIFTS_M:
                corrected, ik_metrics, z_offset_m = _correct_episode(
                    model,
                    episode,
                    pose,
                    reference_grasp_z_m,
                    placement_lift_m,
                )
                table = robot_table_metrics(
                    model, corrected, episode.bindings, pose
                )
                if (
                    ik_metrics["max_position_error_m"] > IK_TOLERANCE_M
                    or table["contact_frames"] != 0
                ):
                    continue
                last_table_safe = (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    z_offset_m,
                    table,
                )
                box, _plan, box_metrics = find_collision_free_box_pose(
                    model,
                    corrected,
                    episode.bindings,
                    pose,
                    episode.center_frame,
                    dice_xy_offset_m=episode.dice_xy_offset_m,
                    yaw_zero_only=True,
                )
                selected = (*last_table_safe, box, box_metrics)
                if box_metrics["box_wall_contacts"] == 0:
                    break
            if (
                selected is not None
                and selected[-1]["box_wall_contacts"] != 0
            ):
                (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    z_offset_m,
                    table,
                    _box,
                    _box_metrics,
                ) = selected
                box, _plan, box_metrics = find_collision_free_box_pose(
                    model,
                    corrected,
                    episode.bindings,
                    pose,
                    episode.center_frame,
                    dice_xy_offset_m=episode.dice_xy_offset_m,
                )
                selected = (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    z_offset_m,
                    table,
                    box,
                    box_metrics,
                )
            elif selected is None and last_table_safe is not None:
                (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    z_offset_m,
                    table,
                ) = last_table_safe
                box, _plan, box_metrics = find_collision_free_box_pose(
                    model,
                    corrected,
                    episode.bindings,
                    pose,
                    episode.center_frame,
                    dice_xy_offset_m=episode.dice_xy_offset_m,
                )
                selected = (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    z_offset_m,
                    table,
                    box,
                    box_metrics,
                )
            if selected is None:
                record["reason"] = "No IK-valid trajectory clears the table"
            else:
                (
                    placement_lift_m,
                    _corrected,
                    ik_metrics,
                    z_offset_m,
                    table,
                    box,
                    box_metrics,
                ) = selected
                status = "ok" if box_metrics["box_wall_contacts"] == 0 else "failed"
                record.update(
                    {
                        "status": status,
                        "right_arm_z_offset_m": z_offset_m,
                        "placement_lift_m": placement_lift_m,
                        "ik": ik_metrics,
                        "table": table,
                        "box": box,
                        "box_metrics": box_metrics,
                    }
                )
                if status != "ok":
                    record["reason"] = "No collision-free box pose was found"
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return {
        "body_lift_m": body_lift_m,
        "body_pitch_rad": body_pitch_rad,
        "usable": sum(record["status"] == "ok" for record in records),
        "episodes": records,
    }


def _candidate_key(candidate: dict[str, Any], baseline_lift: float, baseline_pitch: float):
    usable = candidate.get("usable", candidate.get("estimated_usable", 0))
    height_offset = abs(candidate.get("reference_height_offset_m", 0.0))
    pose_distance = abs(candidate["body_lift_m"] - baseline_lift) + abs(
        candidate["body_pitch_rad"] - baseline_pitch
    )
    return (-usable, height_offset, pose_distance)


def main() -> None:
    args = parse_args()
    if args.pitch_step_rad <= 0:
        raise ValueError("pitch-step-rad must be positive")
    if args.pitch_min_rad > args.pitch_max_rad:
        raise ValueError("pitch-min-rad must not exceed pitch-max-rad")
    if args.height_offset_step_m <= 0:
        raise ValueError("height-offset-step-m must be positive")
    if args.height_offset_min_m > args.height_offset_max_m:
        raise ValueError("height-offset-min-m must not exceed height-offset-max-m")
    if args.verify_top <= 0:
        raise ValueError("verify-top must be positive")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    episodes = load_episode_inputs(model, dataset_dir, args.max_episodes)
    reference = episodes[0]
    baseline_pose = fixed_upper_body_pose(
        args.baseline_body_lift_m,
        args.baseline_body_pitch_rad,
    )
    target_z_m = float(
        _right_gripper_center_at_frame(
            model,
            reference.trajectory,
            reference.bindings,
            baseline_pose,
            reference.center_frame,
        )[2]
    )

    pitch_values = np.arange(
        args.pitch_min_rad,
        args.pitch_max_rad + 0.5 * args.pitch_step_rad,
        args.pitch_step_rad,
    )
    if args.pitch_min_rad <= args.baseline_body_pitch_rad <= args.pitch_max_rad:
        pitch_values = np.unique(
            np.append(pitch_values, args.baseline_body_pitch_rad)
        )
    height_offsets = np.arange(
        args.height_offset_min_m,
        args.height_offset_max_m + 0.5 * args.height_offset_step_m,
        args.height_offset_step_m,
    )
    coarse = []
    total_candidates = len(pitch_values) * len(height_offsets)
    candidate_index = 0
    for height_offset_m in height_offsets:
        for pitch in pitch_values:
            candidate_index += 1
            lift, actual_z = solve_body_lift_for_reference_height(
                model,
                reference,
                float(pitch),
                target_z_m + float(height_offset_m),
                args.reference_height_tolerance_m,
            )
            print(
                f"[{candidate_index}/{total_candidates}] coarse "
                f"lift={lift:.6f} m, pitch={pitch:.6f} rad, "
                f"height_offset={height_offset_m:.4f} m",
                flush=True,
            )
            candidate = coarse_candidate_score(
                model,
                episodes,
                lift,
                float(pitch),
                actual_z,
            )
            candidate["reference_grasp_z_m"] = actual_z
            candidate["reference_height_offset_m"] = float(height_offset_m)
            print(f"    estimated usable={candidate['estimated_usable']}", flush=True)
            coarse.append(candidate)

    coarse.sort(
        key=lambda candidate: _candidate_key(
            candidate,
            args.baseline_body_lift_m,
            args.baseline_body_pitch_rad,
        )
    )
    verified = []
    if not args.coarse_only:
        for index, candidate in enumerate(coarse[: args.verify_top], start=1):
            print(
                f"[{index}/{min(args.verify_top, len(coarse))}] full verify "
                f"lift={candidate['body_lift_m']:.6f} m, "
                f"pitch={candidate['body_pitch_rad']:.6f} rad",
                flush=True,
            )
            result = verified_candidate_score(
                model,
                episodes,
                candidate["body_lift_m"],
                candidate["body_pitch_rad"],
                candidate["reference_grasp_z_m"],
            )
            result["reference_grasp_z_m"] = candidate["reference_grasp_z_m"]
            result["reference_height_offset_m"] = candidate[
                "reference_height_offset_m"
            ]
            print(f"    verified usable={result['usable']}", flush=True)
            verified.append(result)
        verified.sort(
            key=lambda candidate: _candidate_key(
                candidate,
                args.baseline_body_lift_m,
                args.baseline_body_pitch_rad,
            )
        )

    recommended = verified[0] if verified else coarse[0]
    document = {
        "schema": REPORT_SCHEMA,
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "baseline": {
            "body_lift_m": args.baseline_body_lift_m,
            "body_pitch_rad": args.baseline_body_pitch_rad,
            "reference_grasp_z_m": target_z_m,
        },
        "search": {
            "pitch_min_rad": args.pitch_min_rad,
            "pitch_max_rad": args.pitch_max_rad,
            "pitch_step_rad": args.pitch_step_rad,
            "height_offset_min_m": args.height_offset_min_m,
            "height_offset_max_m": args.height_offset_max_m,
            "height_offset_step_m": args.height_offset_step_m,
            "episodes": len(episodes),
            "verify_top": 0 if args.coarse_only else args.verify_top,
        },
        "recommended": recommended,
        "coarse_candidates": coarse,
        "verified_candidates": verified,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    score = recommended.get("usable", recommended.get("estimated_usable"))
    print(f"Report: {output_path}")
    print(
        f"Recommended: body_lift_m={recommended['body_lift_m']:.6f}, "
        f"body_pitch_rad={recommended['body_pitch_rad']:.6f}, "
        f"usable={score}/{len(episodes)}"
    )


if __name__ == "__main__":
    main()
