# A2D MuJoCo replay

使用 `assets/A2D_Omnipicker/A2D.urdf` 在原生 MuJoCo 中回放
`datasets/fixed_spine3_to_g1` 的双臂 `action_joint_position`。

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
MuJoCo `qpos`。这是运动学回放，不经过控制器或动力学跟踪。

常用选项：

```bash
# 只播放一次，或以两倍速度播放
.venv/bin/python scripts/replay_a2d.py --no-loop --speed 2

# 隐藏末端目标，或显示碰撞几何
.venv/bin/python scripts/replay_a2d.py --hide-target --show-collision

# 不打开窗口，运行 FK 诊断
.venv/bin/python scripts/replay_a2d.py --headless
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

A2D URDF 将每侧四连杆夹爪写成八个独立树关节，未描述闭环或 mimic 约束；轨迹
也没有夹爪通道。为避免可视化时机构被拉散，所有脚本都将夹爪保持在 URDF 零位。

## 测试

```bash
.venv/bin/python -m pytest -q
```
