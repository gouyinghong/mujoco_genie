# 独立 PICO 到 G1 数据处理流水线

本文档对应目录：

```text
/home/gyh/mujoco_genie/pico_to_g1_pipeline
```

该目录是从 `/home/gyh/vla_deploy` 迁移出的独立数据处理环境。源项目代码和
数据仍然保留，迁移过程没有移动或删除源文件。

## 1. 数据安全约束

目标项目已有处理完成的数据：

```text
/home/gyh/mujoco_genie/datasets
```

该目录是受保护目录。本流水线遵守以下规则：

- 原始数据放在 `pico_to_g1_pipeline/data`；
- 中间 LeRobot 数据放在 `pico_to_g1_pipeline/work`；
- 新 retarget 结果放在 `pico_to_g1_pipeline/outputs`；
- 不向 `/home/gyh/mujoco_genie/datasets` 写入；
- 各写入脚本包含路径保护，目标位于受保护目录时直接报错；
- 一键入口遇到已存在的阶段输出时直接停止，不自动覆盖；
- 不使用 `rsync --delete`，也不自动删除历史输出。

只有经过人工检查并明确决定发布时，才应另行将新结果复制到目标项目的
`datasets`，而且应使用新的目录名。

## 2. 目录结构

```text
pico_to_g1_pipeline/
├── .venv/                         # 本流水线独立 Python 环境
├── assets/
│   └── G1_120s/                   # 与源流程一致的 G1 URDF
├── data/
│   ├── raw/
│   │   └── session_20260723_084253_461/
│   └── reference/
│       └── genie1_pick_up_dice_804/
├── ego2robot/
│   ├── adjust_0723_gripper_trajectory.py
│   ├── adjust_single_gripper_trajectory_in_place.py
│   ├── estimate_spine3_to_g1_mapping.py
│   ├── retarget_fixed_spine3_to_g1.py
│   └── retarget_spine3_to_g1.py
├── manifests/                     # 迁移校验清单
├── outputs/                       # 最终 G1 retarget 输出
├── work/                          # LeRobot 中间数据
├── convert_pico_session_direct_to_lerobot.py
├── split_human_lerobot_episode.py
├── frame_range.json
├── pipeline_safety.py
├── requirements.txt
├── run_pipeline.py
├── verify_pipeline.py
└── readme_for_dara.md
```

## 3. 环境准备

进入流水线目录：

```bash
cd /home/gyh/mujoco_genie/pico_to_g1_pipeline
```

创建独立环境：

```bash
/home/gyh/.local/bin/uv venv --python /usr/bin/python3 .venv
/home/gyh/.local/bin/uv pip install \
  --python .venv/bin/python \
  -r requirements.txt
```

本机系统 Python 没有安装 `python3.10-venv`，因此这里使用已有的 `uv` 创建
环境，不要求修改系统包。当前 `.venv` 已经创建并安装好依赖，通常无需重复执行。

主要固定版本：

```text
numpy==2.2.6
scipy==1.15.3
pyarrow==25.0.0
lerobot==0.4.4
av==15.1.0
pin==4.1.0
```

系统还需要提供：

```text
ffmpeg
ffprobe
```

检查输入文件和依赖：

```bash
./.venv/bin/python verify_pipeline.py
```

## 4. 总体数据流程

```text
原始 PICO session
  data/raw/session_20260723_084253_461
        │
        │ convert_pico_session_direct_to_lerobot.py
        ▼
单 episode LeRobot v3
  work/lerobot_session_20260723_084253_461_self_contained
        │
        │ split_human_lerobot_episode.py + frame_range.json
        ▼
31 episode LeRobot v3
  work/lerobot_session_20260723_084253_461_split_self_contained
        │
        │ ego2robot/retarget_fixed_spine3_to_g1.py
        ▼
G1 关节与夹爪轨迹
  outputs/fixed_spine3_to_g1_0723_complete
```

## 5. 一键运行

先只检查命令和写入路径，不执行：

```bash
./.venv/bin/python run_pipeline.py --dry-run
```

运行全部阶段：

```bash
./.venv/bin/python run_pipeline.py
```

也可以单独运行阶段：

```bash
./.venv/bin/python run_pipeline.py --stage convert
./.venv/bin/python run_pipeline.py --stage split
./.venv/bin/python run_pipeline.py --stage retarget
```

注意：如果对应输出已经存在，一键入口会拒绝覆盖。需要保留旧结果并使用新目录，
或者人工确认后处理旧的工作目录。

## 6. 阶段一：原始 PICO 转 LeRobot

手动命令：

```bash
./.venv/bin/python convert_pico_session_direct_to_lerobot.py \
  data/raw/session_20260723_084253_461 \
  --output-path work/lerobot_session_20260723_084253_461_self_contained \
  --downsample-factor 1 \
  --sg-window 0
```

转换器完成：

- 跟踪帧与双目视频帧的时间同步；
- 左右目视频拆分；
- 左右手相对当前 SPINE3 的 `action_eef`；
- `action_delta_eef`；
- 原始 PICO `hand_status`；
- 原始手部跟踪有效性；
- 每帧 `spine3_world_xyzw`；
- `source_frame_index` 与相机帧索引。

推荐保持：

```text
--sg-window 0
```

不在完整长序列上平滑人手 EEF。最终 G1 关节会在 episode 切分后独立平滑。

### 6.1 原始 hand_status

`hand_status` 根据拇指尖和食指尖距离计算：

```text
0.02 m -> 0.0
0.14 m -> 1.0
```

该信号可能受 PICO 手部跟踪误差影响，因此最终不会直接作为默认机器人夹爪命令。

### 6.2 spine3_world_xyzw

```text
[x, y, z, qx, qy, qz, qw]
```

该字段让 split 数据集包含固定 SPINE3 变换所需的全部信息，因此 retarget
阶段不再需要访问原始 `body_tracking.jsonl`。

## 7. 阶段二：切分 episode

手动命令：

```bash
./.venv/bin/python split_human_lerobot_episode.py \
  --src-root work/lerobot_session_20260723_084253_461_self_contained \
  --output-root work/lerobot_session_20260723_084253_461_split_self_contained \
  --ranges-file frame_range.json
```

`frame_range.json` 使用闭区间：

```text
[start_frame_idx, end_frame_idx]
```

0723 数据预期结果：

```text
输入：1 episode / 9428 frames
输出：31 episodes / 3357 frames
```

splitter 会自动保留 `spine3_world_xyzw` 及其他自定义字段，并将每个新 episode
第一帧的 `action_delta_eef` 重置为零。

## 8. 阶段三：固定 SPINE3 并 Retarget 到 G1

手动命令：

```bash
./.venv/bin/python ego2robot/retarget_fixed_spine3_to_g1.py \
  --human-root work/lerobot_session_20260723_084253_461_split_self_contained \
  --robot-root data/reference/genie1_pick_up_dice_804 \
  --urdf assets/G1_120s/G1_120s.urdf \
  --output-dir outputs/fixed_spine3_to_g1_0723_complete \
  --all-episodes
```

脚本执行：

1. 使用 split 数据中的 `spine3_world_xyzw`；
2. 将每帧移动 SPINE3 下的手部位姿变换到 episode 0/frame 0 的固定 SPINE3；
3. 使用机器人参考数据估计人类空间到 G1 空间的统计映射；
4. 使用复制过来的精确 G1 URDF 计算双臂 IK；
5. 按 episode 平滑 `action_joint_position`；
6. 根据右手抓取、放置、返回轨迹生成夹爪命令；
7. 写入 episode 报告、映射和 summary。

### 8.1 固定 SPINE3

```text
T_S0_H(t) = inverse(T_W_S0) * T_W_S(t) * T_S(t)_H(t)
```

生成：

```text
source_action_eef_spine3_xyzw
human_eef_fixed_spine3_xyzw
```

### 8.2 G1 关节平滑

默认：

```text
--joint-smooth-window 11
--joint-smooth-polyorder 2
--joint-smooth-passes 1
```

首帧和末帧保持不变。关闭方式：

```bash
--joint-smooth-window 0
```

平滑后不会重新运行 FK，因此 EEF 误差字段仍对应平滑前 IK 结果；关节回放以
`action_joint_position` 为准。

### 8.3 自动夹爪

默认根据 `target_eef_wxyz` 的右手高度生成：

```text
张开 -> 抓取前渐变闭合 -> 搬运/放置保持 -> 返回时渐变张开
```

默认参数：

```text
object_width = 0.060 m
gripper_min_width = 0.035 m
gripper_max_width = 0.120 m
grasp_command = 0.2941176
```

关闭自动夹爪：

```bash
--disable-auto-gripper
```

最终同时保存：

| 字段 | 含义 |
| --- | --- |
| `source_hand_status` | 原始 PICO 手指距离开合度 |
| `source_hand_status_valid` | 原始 PICO 跟踪有效性 |
| `hand_status` | 自动生成的夹爪轨迹 |
| `action_effector` | 机器人使用的夹爪命令 |

## 9. 修正异常夹爪 episode

先预览：

```bash
./.venv/bin/python ego2robot/adjust_single_gripper_trajectory_in_place.py \
  --input-root outputs/fixed_spine3_to_g1_0723_complete \
  --episode-idx 12 \
  --dry-run
```

手动覆盖边界：

```bash
./.venv/bin/python ego2robot/adjust_single_gripper_trajectory_in_place.py \
  --input-root outputs/fixed_spine3_to_g1_0723_complete \
  --episode-idx 12 \
  --close-start-frame 25 \
  --close-end-frame 35 \
  --open-start-frame 70 \
  --open-end-frame 80
```

单集脚本会拒绝修改 `/home/gyh/mujoco_genie/datasets` 中的文件。

## 10. 输出验证

转换后确认：

- `meta/info.json` 中包含 `spine3_world_xyzw`；
- 长序列为 9428 帧。

切分后确认：

- 31 个 episode；
- 3357 帧；
- `meta/split_report.json` 存在。

retarget 后确认：

- 每个 episode 同时存在 NPZ 和 report；
- `action_joint_position` 无 NaN/Inf；
- report 包含 `joint_smoothing`；
- report 包含 `gripper_adjustment`；
- `source_hand_status` 和 `action_effector` 同时存在；
- 左夹爪始终为 1；
- 默认右夹爪抓取值约为 0.2941176；
- `retarget_summary.json` 存在。

## 11. Smoke test

短轨迹不能覆盖完整抓取—放置—返回过程，因此关闭自动夹爪：

```bash
./.venv/bin/python ego2robot/retarget_fixed_spine3_to_g1.py \
  --human-root work/lerobot_session_20260723_084253_461_split_self_contained \
  --episode-idx 0 \
  --max-frames 20 \
  --disable-auto-gripper \
  --output-dir /tmp/fixed_spine3_smoke
```

如果少于 11 帧，同时传入：

```bash
--joint-smooth-window 0
```

## 12. 已迁移内容

已复制而非移动：

- 原始 PICO session；
- `genie1_pick_up_dice_804` 机器人参考数据；
- 当前流程实际使用的 G1_120s URDF资产；
- 当前工作树中的最新版转换、切分、retarget、平滑和夹爪代码；
- `frame_range.json`。

目标项目原有 `datasets` 和 `assets/G1_120s` 均未覆盖。
