#!/usr/bin/env python3
"""将 SPINE3 坐标系下的人体双手末端位姿重映射为 G1 双臂关节轨迹。

用法：
  ./.venv/bin/python ego2robot/retarget_spine3_to_g1.py --episode-idx 0
  ./.venv/bin/python ego2robot/retarget_spine3_to_g1.py --all-episodes

运行前需要先使用 estimate_spine3_to_g1_mapping.py 生成映射参数。
本脚本输出 NPZ 轨迹和 JSON 报告，不会修改原始 LeRobot 数据集。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

try:
    import pinocchio as pin
except ImportError as exc:
    raise SystemExit("需要安装 Pinocchio；请使用 ./.venv/bin/python 运行本脚本") from exc


# 输入、映射、URDF 和输出的默认路径。
# 脚本位于 <项目根目录>/ego2robot/，因此项目资源需要从父目录查找；
# 映射文件和重映射输出则保存在脚本所在的 ego2robot 目录中。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_safety import refuse_protected_dataset_write

DEFAULT_HUMAN_ROOT = (
    PROJECT_ROOT / "work/lerobot_session_20260723_084253_461_split_self_contained"
)
DEFAULT_ROBOT_ROOT = PROJECT_ROOT / "data/reference/genie1_pick_up_dice_804"
DEFAULT_MAPPING = SCRIPT_DIR / "spine3_to_g1_mapping.json"
DEFAULT_URDF = PROJECT_ROOT / "assets/G1_120s/G1_120s.urdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/spine3_to_g1"

# IK 和 FK 使用的机器人参考坐标系与左右末端 link。
BASE_FRAME = "arm_base_link"
LEFT_END_FRAME = "arm_l_end_link"
RIGHT_END_FRAME = "arm_r_end_link"
# 输出的 14 维关节顺序固定为：左臂 joint1～7，然后右臂 joint1～7。
LEFT_ARM_JOINTS = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


def parse_args() -> argparse.Namespace:
    """解析输入数据、IK 权重、成功阈值和 episode 选择参数。"""
    parser = argparse.ArgumentParser(description="将人体 SPINE3 末端轨迹重映射到 G1 双臂")
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN_ROOT)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT,
                        help="机器人参考数据集，用于 IK 初始姿态和 fixed 左臂 joint/EEF")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episode-idx", type=int, default=0)
    selection.add_argument("--all-episodes", action="store_true")
    # fixed：左臂保持机器人示范中的固定姿态；mapped：左臂也跟随人体左手。
    parser.add_argument("--left-mode", choices=("fixed", "mapped"), default="fixed")
    parser.add_argument(
        "--fixed-left-reference-index",
        type=int,
        default=0,
        help="fixed 模式复制机器人数据的全局帧下标，默认 0",
    )
    # 四项权重依次控制位置、方向、帧间连续性和自然姿态。
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.001)
    parser.add_argument("--posture-weight", type=float, default=0.0001)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--position-success-m", type=float, default=0.03)
    parser.add_argument("--orientation-success-deg", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="每个 episode 的调试帧数上限；0 表示处理全部帧")
    args = parser.parse_args()
    refuse_protected_dataset_write(args.output_dir, purpose="write retarget output")
    if args.episode_idx is not None and args.episode_idx < 0:
        parser.error("--episode-idx must be non-negative")
    if args.fixed_left_reference_index < 0:
        parser.error("--fixed-left-reference-index 不能为负数")
    for name in ("position_weight", "orientation_weight", "smoothness_weight", "posture_weight"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_nfev < 1 or args.max_frames < 0:
        parser.error("--max-nfev must be positive and --max-frames must be non-negative")
    return args


def read_parquet_columns(root: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """读取 LeRobot 数据集中所有 Parquet 文件的指定字段。

    List 类型字段转换为二维 NumPy 数组，标量字段转换为一维数组。
    数据没有在这里按 episode 筛选，筛选工作由 ``retarget_episode`` 完成。
    """
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"在 {root / 'data'} 下没有找到 Parquet 文件")
    # 多个 chunk/file 的数据按文件名顺序合并为一张逻辑表。
    tables = [pq.read_table(path, columns=columns) for path in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    result: dict[str, np.ndarray] = {}
    for name in columns:
        column = table[name]
        if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
            result[name] = np.asarray(column.to_pylist())
        else:
            result[name] = np.asarray(column.to_numpy())
    return result


def load_mapping(path: Path) -> dict[str, Any]:
    """读取映射 JSON，并检查其版本是否与本脚本兼容。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到映射 JSON: {path}。请先运行 estimate_spine3_to_g1_mapping.py。"
        )
    with path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    if mapping.get("schema") != "spine3_to_g1_statistical_mapping.v1":
        raise ValueError(f"{path} 使用了不支持的映射格式: {mapping.get('schema')!r}")
    return mapping


def rotation_from_wxyz(quaternion: list[float] | np.ndarray) -> Rotation:
    """把机器人使用的 wxyz 四元数转换为 SciPy 使用的 xyzw。"""
    q = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def pose_from_transform(transform: Any) -> np.ndarray:
    """把 Pinocchio SE3 转换为 xyz+qwqxqyqz，并统一四元数符号。"""
    q_xyzw = np.asarray(pin.Quaternion(transform.rotation).coeffs(), dtype=np.float64)
    if q_xyzw[3] < 0:
        q_xyzw = -q_xyzw
    return np.r_[transform.translation, q_xyzw[[3, 0, 1, 2]]]


class G1ArmIK:
    """基于 G1 URDF 的双臂正运动学和逐侧逆运动学求解器。"""

    def __init__(self, urdf: Path) -> None:
        if not urdf.is_file():
            raise FileNotFoundError(f"找不到 URDF: {urdf}")
        # Pinocchio 模型包含完整机器人；优化时只写入双臂 14 个关节。
        self.urdf = urdf.resolve()
        self.model = pin.buildModelFromUrdf(str(self.urdf))
        self.data = self.model.createData()
        self.neutral = pin.neutral(self.model)
        self.base_frame_id = self._frame_id(BASE_FRAME)
        self.end_frame_ids = {
            "left": self._frame_id(LEFT_END_FRAME),
            "right": self._frame_id(RIGHT_END_FRAME),
        }
        self.arm_indices = {
            "left": np.asarray([self._joint_index(name) for name in LEFT_ARM_JOINTS]),
            "right": np.asarray([self._joint_index(name) for name in RIGHT_ARM_JOINTS]),
        }
        all_indices = np.r_[self.arm_indices["left"], self.arm_indices["right"]]
        # 从 URDF 中读取关节上下限，作为 least_squares 的硬边界。
        self.lower = self.model.lowerPositionLimit[all_indices].copy()
        self.upper = self.model.upperPositionLimit[all_indices].copy()

    def _joint_index(self, name: str) -> int:
        """查询一自由度关节在 Pinocchio 配置向量 q 中的下标。"""
        joint_id = self.model.getJointId(name)
        if joint_id == 0 or self.model.joints[joint_id].nq != 1:
            raise ValueError(f"URDF 中缺少一自由度关节: {name}")
        return self.model.joints[joint_id].idx_q

    def _frame_id(self, name: str) -> int:
        """查询 link 对应的 Pinocchio frame ID。"""
        frame_id = self.model.getFrameId(name)
        if frame_id >= self.model.nframes:
            raise ValueError(f"URDF 中找不到 frame: {name}")
        return frame_id

    def _configuration(self, arm_q: np.ndarray) -> np.ndarray:
        """把 14 维双臂关节角写入完整机器人中性配置。"""
        configuration = self.neutral.copy()
        configuration[self.arm_indices["left"]] = arm_q[:7]
        configuration[self.arm_indices["right"]] = arm_q[7:]
        return configuration

    def frame_transform(self, arm_q: np.ndarray, side: str) -> Any:
        """计算指定末端相对 arm_base_link 的 SE3 变换。"""
        configuration = self._configuration(arm_q)
        pin.forwardKinematics(self.model, self.data, configuration)
        pin.updateFramePlacements(self.model, self.data)
        # oMf 是世界坐标系到 frame 的位姿；左乘 base 的逆得到 base→末端。
        return self.data.oMf[self.base_frame_id].inverse() * self.data.oMf[self.end_frame_ids[side]]

    def forward(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对 14 维关节角进行 FK，返回左右末端 xyz+qwqxqyqz。"""
        left = pose_from_transform(self.frame_transform(arm_q, "left"))
        right = pose_from_transform(self.frame_transform(arm_q, "right"))
        return left, right

    def solve_side(
        self,
        side: str,
        target_position: np.ndarray,
        target_rotation: Rotation,
        arm_q: np.ndarray,
        posture_q: np.ndarray,
        args: argparse.Namespace,
        use_smoothness: bool = True,
    ) -> tuple[np.ndarray, bool, int]:
        """只优化一侧 7 个关节，另一侧保持不变。

        残差由末端位置、末端方向、与上一帧的连续性以及与自然姿态的
        偏差组成。返回更新后的 14 维关节角、优化器状态和函数评估次数。
        """
        side_slice = slice(0, 7) if side == "left" else slice(7, 14)
        # 当前帧以上一帧结果为初值，这是避免 IK 解跳变的主要手段。
        previous = arm_q[side_slice].copy()
        lower = self.lower[side_slice]
        upper = self.upper[side_slice]
        start = np.clip(previous, lower + 1e-8, upper - 1e-8)
        smoothness_weight = args.smoothness_weight if use_smoothness else 0.0

        def residual(side_q: np.ndarray) -> np.ndarray:
            """构造 least_squares 使用的加权残差向量。"""
            candidate = arm_q.copy()
            candidate[side_slice] = side_q
            current = self.frame_transform(candidate, side)
            current_rotation = Rotation.from_matrix(current.rotation)
            position_error = current.translation - target_position
            rotation_error = (current_rotation.inv() * target_rotation).as_rotvec()
            # 权重开平方，因为 least_squares 最终最小化的是残差平方和。
            return np.concatenate(
                (
                    np.sqrt(args.position_weight) * position_error,
                    np.sqrt(args.orientation_weight) * rotation_error,
                    np.sqrt(smoothness_weight) * (side_q - previous),
                    np.sqrt(args.posture_weight) * (side_q - posture_q[side_slice]),
                )
            )

        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=args.max_nfev,
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
        )
        solved = arm_q.copy()
        solved[side_slice] = result.x
        return solved, bool(result.success), int(result.nfev)


def map_side_pose(
    human_pose: np.ndarray,
    side_mapping: dict[str, Any],
    axis_rotation: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    """把一侧人体末端位姿映射到机器人 arm_base_link。

    位置先应用父坐标轴旋转，再进行逐轴缩放和平移：
        p_robot = scale * (R_axis @ p_human) + offset

    姿态以人体平均姿态为中心提取相对旋转，将其幅度乘以 beta 后，
    先在人手局部坐标系中叠加，然后再右乘人手到机器人 TCP 的
    姿态偏置：
        R_target = R_axis * R_human_mean * Exp(beta * Log(delta_h)) * R_offset

    因此 beta=1 时严格退化为
        R_target = R_axis * R_human * R_offset
    而 beta=0 时保持在标定的机器人平均姿态。
    """
    position_mapping = side_mapping["position_mapping"]
    scale = np.asarray(position_mapping["scale_xyz"], dtype=np.float64)
    offset = np.asarray(position_mapping["offset_xyz"], dtype=np.float64)
    position = scale * (axis_rotation @ human_pose[:3]) + offset

    # 人体数据本身保存为 xyzw，正好符合 SciPy 的输入顺序。
    human_rotation = Rotation.from_quat(human_pose[3:7])
    human_mean = Rotation.from_quat(
        side_mapping["human_statistics"]["orientation"]["mean_quaternion_xyzw"]
    )
    orientation_offset = Rotation.from_quat(
        side_mapping["orientation_mapping"]["offset_quaternion_xyzw"]
    )
    beta = float(side_mapping["orientation_mapping"]["relative_motion_scale_recommended"])
    # relative_rotvec 表示当前人体手姿态相对人体平均姿态的旋转。
    relative_rotvec = (human_mean.inv() * human_rotation).as_rotvec()
    axis_rotation_as_rotation = Rotation.from_matrix(axis_rotation)
    scaled_relative_rotation = Rotation.from_rotvec(beta * relative_rotvec)
    target_rotation = (
        axis_rotation_as_rotation
        * human_mean
        * scaled_relative_rotation
        * orientation_offset
    )
    return position, target_rotation


def rotation_error_rad(achieved_wxyz: np.ndarray, target: Rotation) -> float:
    """计算实际姿态到目标姿态的最短旋转角，单位为弧度。"""
    achieved = rotation_from_wxyz(achieved_wxyz)
    return float(np.linalg.norm((achieved.inv() * target).as_rotvec()))


def load_robot_reference(
    robot_root: Path,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed_left_reference_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取机器人 IK 参考姿态和一组配对的固定左臂 joint/EEF。

    ``posture`` 仍由全部机器人关节数据的逐维中位数得到，只用于 IK
    初值和自然姿态约束。固定左臂则直接复制同一机器人数据帧中的
    ``action_joint_position[:7]`` 和 ``action_eef[:7]``，不再计算左臂 IK/FK。
    """
    data = read_parquet_columns(robot_root, ["action_joint_position", "action_eef"])
    joints = np.asarray(data["action_joint_position"], dtype=np.float64)
    eef = np.asarray(data["action_eef"], dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 14:
        raise ValueError(f"机器人 action_joint_position 形状为 {joints.shape}，期望 (N, 14)")
    if eef.shape != joints.shape:
        raise ValueError(f"机器人 action_eef 形状为 {eef.shape}，期望 {joints.shape}")
    if not np.isfinite(joints).all() or not np.isfinite(eef).all():
        raise ValueError("机器人 action_joint_position 或 action_eef 中存在 NaN/Inf")
    if fixed_left_reference_index >= len(joints):
        raise ValueError(
            f"--fixed-left-reference-index={fixed_left_reference_index} 超出机器人数据帧数 {len(joints)}"
        )

    # 中位数比均值更不容易受到少量异常关节值影响。
    posture = np.median(joints, axis=0)
    posture = np.clip(posture, lower + 1e-6, upper - 1e-6)

    # joint 和 EEF 必须取自同一帧，保证它们在原机器人数据中是配对的。
    fixed_left_joint = joints[fixed_left_reference_index, :7].copy()
    fixed_left_eef = eef[fixed_left_reference_index, :7].copy()
    if np.any(fixed_left_joint < lower[:7]) or np.any(fixed_left_joint > upper[:7]):
        raise ValueError("固定左臂参考关节角超出 URDF 关节限位")
    quaternion_norm = np.linalg.norm(fixed_left_eef[3:7])
    if quaternion_norm < 1e-8:
        raise ValueError("固定左臂参考 EEF 包含零长度四元数")
    fixed_left_eef[3:7] /= quaternion_norm
    return posture, fixed_left_joint, fixed_left_eef


def retarget_episode(
    episode_idx: int,
    source: dict[str, np.ndarray],
    mapping: dict[str, Any],
    ik: G1ArmIK,
    posture: np.ndarray,
    fixed_left_joint: np.ndarray,
    fixed_left_eef: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """重映射一个 episode，并返回轨迹、FK 结果和误差诊断。"""
    # 从已经一次性加载的全数据中筛选目标 episode，再按 frame_index 排序。
    mask = source["episode_index"].astype(np.int64) == episode_idx
    if not np.any(mask):
        raise ValueError(f"在 {args.human_root} 中找不到 episode {episode_idx}")
    order = np.argsort(source["frame_index"][mask].astype(np.int64))
    human_eef = np.asarray(source["action_eef"][mask][order], dtype=np.float64)
    if args.max_frames:
        human_eef = human_eef[: args.max_frames]

    frame_count = len(human_eef)
    # 预先分配输出数组，避免在逐帧循环中不断扩展列表。
    joint_positions = np.empty((frame_count, 14), dtype=np.float64)
    target_eef = np.empty((frame_count, 14), dtype=np.float64)
    achieved_eef = np.empty((frame_count, 14), dtype=np.float64)
    position_error = np.empty((frame_count, 2), dtype=np.float64)
    orientation_error = np.empty((frame_count, 2), dtype=np.float64)
    optimizer_success = np.empty((frame_count, 2), dtype=np.bool_)
    optimizer_nfev = np.empty((frame_count, 2), dtype=np.int32)

    # 每个 episode 都从同一机器人参考姿态开始，后续帧以上一帧解热启动。
    arm_q = posture.copy()
    axis_rotation = np.asarray(
        mapping["spine3_to_arm_base_axis_rotation"], dtype=np.float64
    )
    if axis_rotation.shape != (3, 3):
        raise ValueError("spine3_to_arm_base_axis_rotation 的形状必须为 (3, 3)")
    if args.left_mode == "fixed":
        # fixed 模式直接复制机器人同一参考帧中的左臂 joint 和 EEF；
        # 不求左臂 IK，也不通过 FK 重新生成左臂 EEF。
        arm_q[:7] = fixed_left_joint
        fixed_left_position = fixed_left_eef[:3]
        fixed_left_rotation = rotation_from_wxyz(fixed_left_eef[3:7])

    for index, source_pose in enumerate(human_eef):
        # 默认左臂固定、右臂跟随；mapped 模式下左右臂均跟随人体。
        if args.left_mode == "fixed":
            # 每帧直接复用机器人参考帧的左臂关节和末端位姿。
            arm_q[:7] = fixed_left_joint
            left_position, left_rotation = fixed_left_position, fixed_left_rotation
            left_ok = True
            left_nfev = 0
        else:
            left_position, left_rotation = map_side_pose(
                source_pose[:7], mapping["left"], axis_rotation
            )
            # 只有 mapped 模式才根据人体左手目标逐帧求解左臂 IK。
            arm_q, left_ok, left_nfev = ik.solve_side(
                "left", left_position, left_rotation, arm_q, posture, args,
                use_smoothness=index > 0,
            )
        right_position, right_rotation = map_side_pose(
            source_pose[7:], mapping["right"], axis_rotation
        )

        # 右臂始终跟随人体右手，所以每一帧都需要求解 IK。
        # 首帧没有“上一帧”，故不使用平滑项，避免初始姿态拉偏目标。
        arm_q, right_ok, right_nfev = ik.solve_side(
            "right", right_position, right_rotation, arm_q, posture, args,
            use_smoothness=index > 0,
        )
        if args.left_mode == "fixed":
            # 固定左臂的 joint/EEF 已经从机器人同一数据帧直接复制；这里只对
            # 右臂 IK 结果做 FK。左臂不再进行额外的 IK 或 FK 计算。
            achieved_left = fixed_left_eef
            achieved_right = pose_from_transform(ik.frame_transform(arm_q, "right"))
        else:
            # mapped 模式的左右臂都由 IK 得到，因此对两侧做 FK 回算。
            achieved_left, achieved_right = ik.forward(arm_q)

        # fixed 模式保持机器人参考帧四元数原值；mapped 模式才把 SciPy
        # 输出的 xyzw 转换为机器人字段使用的 qwxyz。
        left_q = (
            fixed_left_eef[3:7]
            if args.left_mode == "fixed"
            else left_rotation.as_quat()[[3, 0, 1, 2]]
        )
        right_q = right_rotation.as_quat()[[3, 0, 1, 2]]
        joint_positions[index] = arm_q
        target_eef[index] = np.r_[left_position, left_q, right_position, right_q]
        achieved_eef[index] = np.r_[achieved_left, achieved_right]
        position_error[index] = (
            np.linalg.norm(achieved_left[:3] - left_position),
            np.linalg.norm(achieved_right[:3] - right_position),
        )
        orientation_error[index] = (
            rotation_error_rad(achieved_left[3:7], left_rotation),
            rotation_error_rad(achieved_right[3:7], right_rotation),
        )
        optimizer_success[index] = (left_ok, right_ok)
        optimizer_nfev[index] = (left_nfev, right_nfev)

    # 优化器返回成功并不代表误差一定足够小，因此还要检查位置和方向阈值。
    ik_success = (
        optimizer_success
        & (position_error <= args.position_success_m)
        & (orientation_error <= np.deg2rad(args.orientation_success_deg))
    )
    # NPZ 只保留 replay 和 IK 质量检查需要的字段，避免保存重复数据。
    return {
        "episode_idx": episode_idx,
        "target_eef_wxyz": target_eef,
        "action_joint_position": joint_positions,
        "achieved_eef_wxyz": achieved_eef,
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "optimizer_success": optimizer_success,
        "ik_success": ik_success,
        "optimizer_nfev": optimizer_nfev,
    }


def save_episode(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """保存一个 episode 的压缩 NPZ，并生成简要 JSON 误差报告。"""
    episode_idx = int(result["episode_idx"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / f"episode_{episode_idx:06d}.npz"
    np.savez_compressed(
        trajectory_path,
        **{key: value for key, value in result.items() if key != "episode_idx"},
    )
    pos = result["position_error_m"]
    ori_deg = np.degrees(result["orientation_error_rad"])
    report = {
        "episode_idx": episode_idx,
        "frames": int(len(result["action_joint_position"])),
        "trajectory": str(trajectory_path),
        "joint_order": list(ARM_JOINTS),
        "eef_order": "left/right xyz_qwqxqyqz in arm_base_link",
        "position_error_m": {
            "left_mean": float(np.mean(pos[:, 0])),
            "left_max": float(np.max(pos[:, 0])),
            "right_mean": float(np.mean(pos[:, 1])),
            "right_max": float(np.max(pos[:, 1])),
        },
        "orientation_error_deg": {
            "left_mean": float(np.mean(ori_deg[:, 0])),
            "left_max": float(np.max(ori_deg[:, 0])),
            "right_mean": float(np.mean(ori_deg[:, 1])),
            "right_max": float(np.max(ori_deg[:, 1])),
        },
        "ik_success_rate": {
            "left": float(np.mean(result["ik_success"][:, 0])),
            "right": float(np.mean(result["ik_success"][:, 1])),
        },
    }
    report_path = output_dir / f"episode_{episode_idx:06d}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def main() -> int:
    """加载公共资源，处理选定 episode，并写入整体汇总报告。"""
    args = parse_args()
    mapping = load_mapping(args.mapping)
    source = read_parquet_columns(
        args.human_root, ["action_eef", "episode_index", "frame_index"]
    )
    ik = G1ArmIK(args.urdf)
    posture, fixed_left_joint, fixed_left_eef = load_robot_reference(
        args.robot_root,
        ik.lower,
        ik.upper,
        args.fixed_left_reference_index,
    )
    # --all-episodes 使用数据中实际存在的 episode 编号，否则只处理指定编号。
    available = sorted(np.unique(source["episode_index"].astype(np.int64)).tolist())
    episodes = available if args.all_episodes else [args.episode_idx]

    started = time.monotonic()
    reports = []
    for episode_idx in episodes:
        print(f"Retargeting episode {episode_idx}...", flush=True)
        result = retarget_episode(
            episode_idx,
            source,
            mapping,
            ik,
            posture,
            fixed_left_joint,
            fixed_left_eef,
            args,
        )
        report = save_episode(result, args.output_dir)
        reports.append(report)
        print(
            f"  frames={report['frames']} "
            f"right_position_mean={report['position_error_m']['right_mean']:.4f} m "
            f"right_success={report['ik_success_rate']['right']:.1%}",
            flush=True,
        )

    summary = {
        "schema": "spine3_to_g1_retarget.v1",
        "human_dataset": str(args.human_root.resolve()),
        "robot_reference_dataset": str(args.robot_root.resolve()),
        "mapping": str(args.mapping.resolve()),
        "urdf": str(args.urdf.resolve()),
        "left_mode": args.left_mode,
        "fixed_left_reference_index": args.fixed_left_reference_index,
        "weights": {
            "position": args.position_weight,
            "orientation": args.orientation_weight,
            "smoothness": args.smoothness_weight,
            "posture": args.posture_weight,
        },
        "episodes": reports,
        "aggregate": {
            "frames": int(sum(report["frames"] for report in reports)),
            "right_position_error_mean_m": float(
                sum(
                    report["frames"] * report["position_error_m"]["right_mean"]
                    for report in reports
                )
                / sum(report["frames"] for report in reports)
            ),
            "right_position_error_max_m": float(
                max(report["position_error_m"]["right_max"] for report in reports)
            ),
            "right_ik_success_rate": float(
                sum(
                    report["frames"] * report["ik_success_rate"]["right"]
                    for report in reports
                )
                / sum(report["frames"] for report in reports)
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "retarget_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
