#!/usr/bin/env python3
"""根据人体和机器人数据，估计 SPINE3 到 G1 双臂基座的初始重映射参数。

用法：
  ./.venv/bin/python ego2robot/estimate_spine3_to_g1_mapping.py
  ./.venv/bin/python ego2robot/estimate_spine3_to_g1_mapping.py --output mapping.json

人体 action_eef 的格式为 xyz+xyzw，机器人 action_eef 的格式为 xyz+wxyz。
本脚本根据两套同任务数据的统计分布估计初值，不进行逐帧轨迹对齐。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_safety import refuse_protected_dataset_write

# 默认使用已经按 SPINE3 坐标系转换并拆分好的人体 LeRobot 数据集。
DEFAULT_HUMAN_ROOT = (
    PROJECT_ROOT / "work/lerobot_session_20260723_084253_461_split_self_contained"
)
# 机器人参考数据与人体数据执行相同的任务，用于统计 G1 的末端工作范围。
DEFAULT_ROBOT_ROOT = PROJECT_ROOT / "data/reference/genie1_pick_up_dice_804"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并检查上下分位数是否合法。"""
    parser = argparse.ArgumentParser(
        description="根据人体和机器人 LeRobot 数据估计 SPINE3 到 G1 的末端映射。"
    )
    parser.add_argument(
        "--human-root", type=Path, default=DEFAULT_HUMAN_ROOT, help="人体 LeRobot 数据集目录"
    )
    parser.add_argument(
        "--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT, help="机器人 LeRobot 数据集目录"
    )
    parser.add_argument(
        "--low-quantile", type=float, default=0.05, help="稳健范围的下分位数，默认 0.05"
    )
    parser.add_argument(
        "--high-quantile", type=float, default=0.95, help="稳健范围的上分位数，默认 0.95"
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 输出路径")
    args = parser.parse_args()
    if args.output is not None:
        refuse_protected_dataset_write(args.output, purpose="write mapping JSON")
    if not 0.0 <= args.low_quantile < 0.5:
        parser.error("--low-quantile 必须位于 [0, 0.5) 范围内")
    if not 0.5 < args.high_quantile <= 1.0:
        parser.error("--high-quantile 必须位于 (0.5, 1] 范围内")
    return args


def load_action_eef(root: Path) -> np.ndarray:
    """读取数据集所有 Parquet 文件中的 action_eef，并合并为 (N, 14) 数组。

    14 维数据由左手/左末端 7 维和右手/右末端 7 维拼接而成。
    本函数只检查形状和有限值；四元数顺序在 ``split_pose`` 中统一处理。
    """
    # LeRobot v3 数据可能分布在多个 chunk/file Parquet 文件中。
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"在 {root / 'data'} 下没有找到 Parquet 文件")

    # 这里只加载估计映射所需的 action_eef，避免读取视频等无关字段。
    tables = [pq.read_table(path, columns=["action_eef"]) for path in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    values = np.asarray(table["action_eef"].to_pylist(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"{root} 中 action_eef 形状为 {values.shape}，期望 (N, 14)")
    if not np.isfinite(values).all():
        raise ValueError(f"{root} 的 action_eef 中存在 NaN 或无穷大")
    return values


def normalize_quaternions(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """将一批 xyzw 四元数归一化，零长度四元数直接报错。"""
    norms = np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("发现零长度四元数")
    return quaternions_xyzw / norms


def canonical_xyzw(rotation: Rotation) -> np.ndarray:
    """返回规范化符号的 xyzw 四元数，统一令 w >= 0。

    四元数 q 和 -q 表示同一个旋转。统一符号可使 JSON 输出更稳定、便于比较。
    """
    quaternion = rotation.as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """把 SciPy 使用的 xyzw 顺序转换为机器人数据使用的 wxyz 顺序。"""
    return quaternion_xyzw[[3, 0, 1, 2]]


def split_pose(values: np.ndarray, side_offset: int, source: str) -> tuple[np.ndarray, Rotation]:
    """从 14 维 action_eef 中取出一侧的位置和旋转。

    ``side_offset=0`` 表示左侧，``side_offset=7`` 表示右侧。
    人体四元数已经是 SciPy 所需的 xyzw；机器人四元数为 wxyz，需要重排。
    """
    positions = values[:, side_offset : side_offset + 3]
    quaternions = values[:, side_offset + 3 : side_offset + 7]
    if source == "robot":
        # 机器人数据保存顺序为 qw、qx、qy、qz；SciPy 接收 qx、qy、qz、qw。
        quaternions = quaternions[:, [1, 2, 3, 0]]
    quaternions = normalize_quaternions(quaternions)
    return positions, Rotation.from_quat(quaternions)


def position_statistics(positions: np.ndarray, low: float, high: float) -> dict[str, Any]:
    """计算位置均值、标准差以及稳健的低/中/高分位数。"""
    quantiles = np.quantile(positions, [low, 0.5, high], axis=0)
    return {
        "mean_xyz": np.mean(positions, axis=0).tolist(),
        "std_xyz": np.std(positions, axis=0).tolist(),
        "low_xyz": quantiles[0].tolist(),
        "median_xyz": quantiles[1].tolist(),
        "high_xyz": quantiles[2].tolist(),
    }


def orientation_statistics(rotations: Rotation) -> tuple[Rotation, dict[str, Any]]:
    """计算平均旋转以及每帧姿态相对平均旋转的角度分布。"""
    # Rotation.mean() 在旋转流形上求均值，比直接对四元数各分量求平均可靠。
    mean_rotation = rotations.mean()
    # mean^-1 * current 表示当前姿态相对平均姿态的旋转变化。
    angular_deviation = np.linalg.norm((mean_rotation.inv() * rotations).as_rotvec(), axis=1)
    return mean_rotation, {
        "mean_quaternion_xyzw": canonical_xyzw(mean_rotation).tolist(),
        "deviation_deg_median": float(np.degrees(np.quantile(angular_deviation, 0.5))),
        "deviation_deg_p95": float(np.degrees(np.quantile(angular_deviation, 0.95))),
    }


def estimate_side(
    human_values: np.ndarray,
    robot_values: np.ndarray,
    side_offset: int,
    low: float,
    high: float,
) -> dict[str, Any]:
    """估计单侧手臂的位置映射、姿态偏置和运动幅度。

    位置采用逐轴仿射映射：
        p_robot = diag(scale_xyz) @ p_human + offset_xyz

    先让人体与机器人在每个轴上的低/高分位范围一致，再对齐中位数。
    这种方式比直接使用最小值和最大值更不容易受异常帧影响。
    """
    human_position, human_rotation = split_pose(human_values, side_offset, "human")
    robot_position, robot_rotation = split_pose(robot_values, side_offset, "robot")

    # 三行依次对应下分位数、中位数和上分位数。
    human_quantiles = np.quantile(human_position, [low, 0.5, high], axis=0)
    robot_quantiles = np.quantile(robot_position, [low, 0.5, high], axis=0)
    human_span = human_quantiles[2] - human_quantiles[0]
    if np.any(human_span < 1e-8):
        raise ValueError("人体末端位置范围过小，无法估计缩放比例")

    # 先匹配每个轴的稳健运动范围，再通过平移项对齐中位数。
    # scale = robot_span / human_span
    # offset = robot_median - scale * human_median
    scale_xyz = (robot_quantiles[2] - robot_quantiles[0]) / human_span
    offset_xyz = robot_quantiles[1] - scale_xyz * human_quantiles[1]

    human_mean_rotation, human_orientation_stats = orientation_statistics(human_rotation)
    robot_mean_rotation, robot_orientation_stats = orientation_statistics(robot_rotation)

    # 初始假设 SPINE3 与 arm_base_link 的父坐标轴同向：
    # R_robot ≈ R_human * R_offset，因此 R_offset = R_human^-1 * R_robot。
    orientation_offset = human_mean_rotation.inv() * robot_mean_rotation
    human_p95 = human_orientation_stats["deviation_deg_p95"]
    robot_p95 = robot_orientation_stats["deviation_deg_p95"]
    # 用姿态变化 P95 的比值估计机器人应保留多少人体手腕旋转幅度。
    raw_motion_scale = robot_p95 / human_p95 if human_p95 > 1e-8 else 1.0

    return {
        "position_mapping": {
            "formula": "p_robot = diag(scale_xyz) @ p_human + offset_xyz",
            "scale_xyz": scale_xyz.tolist(),
            "offset_xyz": offset_xyz.tolist(),
        },
        "orientation_mapping": {
            "formula": "R_robot = R_human @ R_offset",
            "offset_quaternion_xyzw": canonical_xyzw(orientation_offset).tolist(),
            "offset_quaternion_wxyz": wxyz(canonical_xyzw(orientation_offset)).tolist(),
            "relative_motion_scale_raw": float(raw_motion_scale),
            # 第一版不放大人体旋转，因此将推荐值限制在 [0, 1]。
            "relative_motion_scale_recommended": float(np.clip(raw_motion_scale, 0.0, 1.0)),
        },
        "human_statistics": {
            "position": position_statistics(human_position, low, high),
            "orientation": human_orientation_stats,
        },
        "robot_statistics": {
            "position": position_statistics(robot_position, low, high),
            "orientation": robot_orientation_stats,
        },
        "robot_fixed_pose": {
            # 当某侧机器人手臂在示范中基本不动时，可直接使用该固定末端位姿。
            "position_xyz": np.median(robot_position, axis=0).tolist(),
            "orientation_quaternion_wxyz": wxyz(canonical_xyzw(robot_mean_rotation)).tolist(),
        },
    }


def main() -> None:
    """加载两套数据，分别估计左右手映射，并输出 JSON。"""
    args = parse_args()
    human_values = load_action_eef(args.human_root)
    robot_values = load_action_eef(args.robot_root)

    # action_eef 前 7 维为左侧、后 7 维为右侧，因此偏移分别为 0 和 7。
    result = {
        "schema": "spine3_to_g1_statistical_mapping.v1",
        "method": {
            "position": "Match per-axis robust quantile spans and medians.",
            "orientation": "Match scipy rotation means; scale relative rotation by p95 spread ratio.",
            "warning": "Same-task datasets are not frame paired; use this only as an initial mapping.",
        },
        "human_dataset": str(args.human_root.resolve()),
        "robot_dataset": str(args.robot_root.resolve()),
        "human_frames": int(len(human_values)),
        "robot_frames": int(len(robot_values)),
        "quantiles": [args.low_quantile, 0.5, args.high_quantile],
        # 当前统计表明两个已转换数据集的 XYZ 方向基本一致，初值采用单位旋转。
        "spine3_to_arm_base_axis_rotation": np.eye(3).tolist(),
        "left": estimate_side(
            human_values, robot_values, 0, args.low_quantile, args.high_quantile
        ),
        "right": estimate_side(
            human_values, robot_values, 7, args.low_quantile, args.high_quantile
        ),
        "recommended_first_version": {
            "left_mode": "fixed_robot_pose",
            "right_position_mapping": "use right.position_mapping",
            "right_orientation_mapping": "use right.orientation_mapping with low IK weight",
        },
    }

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        # 输出目录不存在时自动创建；ensure_ascii=False 保留中文可读性。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
