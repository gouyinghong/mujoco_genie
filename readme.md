# G1 MuJoCo replay

`datasets/fixed_spine3_to_g1` 中的双臂重映射数据可以直接在原生 MuJoCo 中回放。

## 快速开始

安装依赖的 Python 环境已经位于 `.venv`。先转换 URDF：

```bash
.venv/bin/python scripts/convert_g1_to_mjcf.py
```

打开交互式 viewer 并循环播放：

```bash
.venv/bin/python scripts/replay_g1.py
```

使用更接近真机外观的 `assets/A2D_Omnipicker/A2D.urdf`：

```bash
.venv/bin/python scripts/replay_g1.py --robot a2d
```

只显示并持续保持轨迹最后一帧（默认使用 A2D）：

```bash
.venv/bin/python scripts/show_last_frame.py

# 显示末端目标点或切回 G1 convex 外观
.venv/bin/python scripts/show_last_frame.py --show-target
.venv/bin/python scripts/show_last_frame.py --robot g1
```

显示双臂 14 个关节全部为零位的静态姿态（默认使用 A2D）：

```bash
.venv/bin/python scripts/show_zero_pose.py

# 同时显示 MuJoCo 中的关节坐标轴，便于和真机检查零位及旋转轴
.venv/bin/python scripts/show_zero_pose.py --show-joint-axes

# 使用 G1 convex 外观
.venv/bin/python scripts/show_zero_pose.py --robot g1
```

首次运行会自动生成 `assets/A2D_Omnipicker/A2D.xml`。也可以手动转换：

```bash
.venv/bin/python scripts/convert_a2d_to_mjcf.py
```

红点为数据中的目标末端位姿，绿点为 `arm_l_end_link` / `arm_r_end_link` 的实际位姿。常用参数：

```bash
# 单次、2 倍速播放
.venv/bin/python scripts/replay_g1.py --no-loop --speed 2

# 隐藏目标点，显示碰撞几何
.venv/bin/python scripts/replay_g1.py --hide-target --show-collision

# 不打开窗口，检查模型、关节限位和 FK 误差
.venv/bin/python scripts/replay_g1.py --headless

# 夹爪完全闭合；1.0 表示完全张开
.venv/bin/python scripts/replay_g1.py --gripper-open 0
```

运行测试：

```bash
.venv/bin/python -m pytest -q
```

转换脚本不会修改源 URDF。它会在内存中修复非法惯量、保留 visual/collision 几何、补充末端调试 site 和 MuJoCo 夹爪 mimic 约束，然后生成：

```text
assets/robot_g1_arms_convex_decomposition/robot_g1_arms_convex_decomposition.xml
```

## 夹爪约定

这个 convex 版本只修改了双臂碰撞体，夹爪控制继承 robot_g1.usda。左右夹爪各控制一个主关节：

- 左夹爪：idx41_gripper_l_outer_joint1
- 右夹爪：idx81_gripper_r_outer_joint1
- 0.0 rad：完全闭合
- 0.785398 rad：完全张开（45°）

内侧关节由 PhysX Mimic 自动反向跟随，不需要单独控制：

- 左内侧：idx31_gripper_l_inner_joint1
- 右内侧：idx71_gripper_r_inner_joint1

MuJoCo MJCF 中由转换脚本显式添加 `inner = -outer` equality 约束。当前 NPZ 不包含逐帧夹爪动作，因此 replay 使用 `--gripper-open` 指定固定开度。

对于 A2D 模型，URDF 将每侧八个四连杆关节写成独立树关节，但没有提供闭环或 mimic 关系。为避免可视化时拉散夹爪，A2D replay 将夹爪保持在 URDF 的零位姿，`--gripper-open` 仅对 G1 convex 模型生效。

## A2D Joint7 零位标定

根据真机零位对比，A2D URDF 已将以下机械零位偏置写入 Joint7 的 `origin`：

- `Joint7_l`: 原仿真 `+pi/2 rad` 对应新的 `0 rad`
- `Joint7_r`: 原仿真 `-pi/2 rad` 对应新的 `0 rad`

偏置采用 `R_new = R_original @ Rz(offset)` 写入，因此 Joint7 的旋转轴和正方向保持不变。修改 URDF 后需要重新生成 A2D MJCF：

```bash
.venv/bin/python scripts/convert_a2d_to_mjcf.py
```
