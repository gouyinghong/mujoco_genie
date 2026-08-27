# A2D MuJoCo replay

使用 `assets/A2D_Omnipicker/A2D.urdf` 在原生 MuJoCo 中回放
`datasets/fixed_spine3_to_g1_add_effector` 的双臂 `action_joint_position`
和左右夹爪 `action_effector`。

## 环境

```bash
.venv/bin/pip install mujoco numpy scipy pytest
```

## 生成 A2D MJCF

机器人头部和躯干的姿态：0, 25.00167804031422, 0.3087556226039414, 0.24924583435058595

```bash
# 默认生成带 140 × 90 × 80 cm 白色桌子和骰子的场景
.venv/bin/python scripts/convert_a2d_to_mjcf.py

# 另存一个不带桌子的机器人版本
.venv/bin/python scripts/convert_a2d_to_mjcf.py --without-table
```

默认输出为 `assets/A2D_Omnipicker/A2D.xml`。回放脚本发现 URDF 比 XML
更新时也会自动重新生成模型。只有机器人的版本输出为
`assets/A2D_Omnipicker/A2D_robot_only.xml`。

## 回放轨迹

```bash
.venv/bin/python scripts/replay_a2d.py

# 回放只有机器人的版本
.venv/bin/python scripts/replay_a2d.py \
  --model assets/A2D_Omnipicker/A2D_robot_only.xml
```

回放直接将数据中的 14 维 `action_joint_position` 按左右臂顺序映射到
`Joint1_l`～`Joint7_l` 和 `Joint1_r`～`Joint7_r`，经过时间戳线性插值后写入
MuJoCo `qpos`。两维 `action_effector` 按 `[左, 右]` 顺序控制四连杆夹爪，
其中 `0` 表示闭合、`1` 表示张开。这是运动学回放，不经过控制器或动力学跟踪。

回放会从 `action_effector` 自动识别主要抓取侧、最后一次持续闭合的起点和随后
释放的时刻。骰子初始化在闭合起点的夹爪中心，闭合期间随夹爪搬运，释放后按
重力轨迹落到桌面。本条轨迹识别为右夹爪第 54 帧开始闭合、第 75 帧释放。

常用选项：

```bash
# 只播放一次，或以两倍速度播放
.venv/bin/python scripts/replay_a2d.py --no-loop --speed 2

# 隐藏末端目标，或显示碰撞几何
.venv/bin/python scripts/replay_a2d.py --hide-target --show-collision

# 不打开窗口，运行 FK 诊断
.venv/bin/python scripts/replay_a2d.py --headless
```

使用 `--no-loop` 时，窗口会停在第 0 帧。先在窗口中调整视角，调整好后让窗口
获得焦点并按空格键开始播放。若希望打开窗口后立即播放，可再加
`--start-immediately`。播放过程中按空格键可以暂停，再按一次则从当前时间点
继续；暂停期间轨迹时间不会前进。

主抓放回放可以用 `--body-lift-m` 指定固定的躯干升降高度。机器人姿态和骰子
初始抓取位置会使用同一个高度重新计算。骰子在抓取前保持水平姿态平放在桌面，
抓取后则保持相对于夹爪的姿态关系。

调整机器人头部和躯干姿态时，可以使用单独的交互回放脚本。它仍会回放双臂和
`action_effector`，但骰子固定在桌面上：

```bash
.venv/bin/python scripts/replay_a2d_torso_adjust.py
```

指定另一套同结构的数据集目录：

```bash
.venv/bin/python scripts/replay_a2d_torso_adjust.py \
  --dataset-dir datasets/fixed_spine3_to_g1_0723_add_effector
```

在 MuJoCo 窗口中用 `W/S` 调节躯干俯仰、`R/F` 调节躯干高度、`I/K` 调节
头部俯仰、`J/L` 调节头部左右角度；按空格开始或重新播放。终端会输出当前的
`[head_yaw_deg, head_pitch_deg, body_pitch_rad, body_lift_m]`，也可以用
`--body-pitch-rad`、`--body-lift-m` 等参数指定初始值。

要让某一侧夹爪在所有帧保持固定开合值，可使用 `--left-effector` 或
`--right-effector`。例如右夹爪完全闭合：

```bash
.venv/bin/python scripts/replay_a2d_torso_adjust.py \
  --dataset-dir datasets/fixed_spine3_to_g1_0723_add_effector \
  --right-effector 0
```

## 静态比较姿态

维持在轨迹最后一帧：

```bash
.venv/bin/python scripts/show_last_frame.py
```

双臂关节全部保持在零位：

```bash
.venv/bin/python scripts/show_zero_pose.py

# 同时显示 MuJoCo 关节轴
.venv/bin/python scripts/show_zero_pose.py --show-joint-axes
```

这两个脚本只做 `mj_forward`，不会调用 `mj_step`，因此机器人不会受重力或接触力
影响而离开指定姿态。

## Joint7 零位标定

`A2D.urdf` 已把真机机械零位固化到 Joint7 的父子坐标变换中：

- `Joint7_l`：仿真旧零位增加 `+π/2 rad`。
- `Joint7_r`：仿真旧零位增加 `-π/2 rad`。

因此数据中的 Joint7 action 不需要在回放脚本里再加偏置。偏置只出现在 URDF
固定变换中，关节变量本身仍保持真机的零位定义和方向。

## 夹爪说明

A2D URDF 将每侧四连杆夹爪写成八个独立树关节，未描述闭环或 mimic 约束。
回放代码依据原始 Omnipicker 的闭环尺寸预先求解联动关系，将 `[0, 1]` 开度映射到
驱动指根关节的 `0～π/4 rad`，并同步设置其余被动关节。没有
`action_effector` 的旧轨迹仍会保持夹爪在 URDF 零位。

## 测试

```bash
.venv/bin/python -m pytest -q
```
