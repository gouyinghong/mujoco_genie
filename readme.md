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

# 保留桌子，只隐藏骰子并关闭骰子碰撞
.venv/bin/python scripts/replay_a2d.py --no-dice

# 骰子水平放在桌面，夹爪闭合完成后再附着
.venv/bin/python scripts/replay_a2d.py --dice-on-table

# 平放骰子，并旋转骰子使侧面与夹爪闭合方向对齐
.venv/bin/python scripts/replay_a2d.py --align-dice-to-gripper

# 使用第 37 帧两侧指尖碰撞几何的中点设置骰子的 x/y
.venv/bin/python scripts/replay_a2d.py \
  --align-dice-to-gripper --dice-center-frame 37

# 使用 assets/objects/box 中的网格和纹理生成独立纸盒场景
.venv/bin/python scripts/convert_a2d_to_mjcf.py --with-cardboard-box
.venv/bin/python scripts/replay_a2d.py \
  --model assets/A2D_Omnipicker/A2D_with_box.xml

# 恢复纸盒原始、未提亮的纹理显示
.venv/bin/python scripts/replay_a2d.py \
  --model assets/A2D_Omnipicker/A2D_with_box.xml \
  --box-texture-gamma 1.0

# 不打开窗口，运行 FK 诊断
.venv/bin/python scripts/replay_a2d.py --headless

.venv/bin/python scripts/replay_a2d.py \
  --episode /home/gyh/mujoco_genie/datasets/fixed_spine3_to_g1_0723_add_effector_after/episode_000000.npz \
  --summary /home/gyh/mujoco_genie/datasets/fixed_spine3_to_g1_0723_add_effector_after/retarget_summary.json \
  --body-lift-m 0.1827884 \
  --speed 0.5 \
  --no-loop
```

使用 `--no-loop` 时，窗口会停在第 0 帧。先在窗口中调整视角，调整好后让窗口
获得焦点并按空格键开始播放。若希望打开窗口后立即播放，可再加
`--start-immediately`。播放过程中按空格键可以暂停，再按一次则从当前时间点
继续；暂停期间轨迹时间不会前进。每次按空格时，终端会同时输出当前轨迹帧、
`episode_frame_index`、原始数据的 `source_frame_index` 和轨迹时间。

主抓放回放可以用 `--body-lift-m` 指定固定的躯干升降高度。机器人姿态和骰子
初始抓取位置会使用同一个高度重新计算。骰子在抓取前保持水平姿态平放在桌面，
抓取后则保持相对于夹爪的姿态关系。

## 固定躯干的整目录回放

可以先搜索一组固定躯干参数，在保持参考抓取高度仅小幅上调（默认最多 20 mm）
的约束下，最大化同时满足 IK、桌面无穿模和纸盒无碰撞的 episode 数量：

```bash
.venv/bin/python scripts/optimize_a2d_fixed_torso.py \
  --dataset-dir datasets/fixed_spine3_to_g1_0723_add_effector_gripper_6cm_return \
  --model assets/A2D_Omnipicker/A2D_with_box.xml
```

搜索报告写入 `datasets/torso_pose_optimization.json`。脚本先做完整俯仰角与参考
高度偏移的快速搜索，再对最佳候选执行骰子、桌面和纸盒的完整碰撞验证。可用
`--pitch-min-rad`、`--pitch-max-rad`、`--pitch-step-rad` 以及
`--height-offset-min-m`、`--height-offset-max-m`、`--height-offset-step-m`
调整搜索范围和精度。

先为目录中的每条 episode 预计算右臂高度校正、骰子初始位置和纸盒位置：

```bash
.venv/bin/python scripts/prepare_a2d_dataset_replay.py \
  --dataset-dir datasets/fixed_spine3_to_g1_0723_add_effector_gripper_6cm_return \
  --body-lift-m 0.264496 \
  --body-pitch-rad 0.387295
```

原始 NPZ 不会被修改。结果写入 `datasets/replay_layouts.json`，校正后的 14 维
关节轨迹写入 `datasets/.replay_cache/<数据集名称>/`。因此替换具体的数据集
子目录不会删除已生成的清单和缓存。预计算会自动跳过不能满足 IK 误差和纸盒
碰撞要求的 episode。

修改 `datasets/replay_layout_overrides.json` 中某一条参数后，可以只重新计算该条
并合并回现有清单，其他 episode 的记录和缓存不会重算：

```bash
.venv/bin/python scripts/prepare_a2d_dataset_replay.py \
  --dataset-dir datasets/fixed_spine3_to_g1_0723_add_effector_gripper_6cm_return \
  --model assets/A2D_Omnipicker/A2D_with_box.xml \
  --body-lift-m 0.264496 \
  --body-pitch-rad 0.387295 \
  --episode episode_000009.npz
```

可重复使用 `--episode` 一次更新多条。首次生成共享清单、修改模型、固定躯干
参数或参考轨迹 `episode_000000.npz` 后，仍需执行一次不带 `--episode` 的完整预处理。

可先无窗口验证所有通过的 episode：

```bash
.venv/bin/python scripts/replay_a2d_dataset.py --headless
```

打开批量可视化窗口：

```bash
.venv/bin/python scripts/replay_a2d_dataset.py --speed 0.5
```

使用真实机器人标定参数，以固定的 1280×800、30 FPS 头部相机画面 replay：

```bash
.venv/bin/python scripts/replay_a2d_head_camera.py --speed 0.5
```

真实标定文件中的 `ppx/ppy` 是从图像左上角开始的绝对像素坐标；脚本会将其
换算为 MuJoCo 要求的、相对图像中心的 `principalpixel` 偏移。默认严格使用
标定的相机位置和方向，不会根据纸盒位置改变相机光轴。

头部相机窗口中：空格暂停/继续，`N/P` 切换 episode，Enter 重新播放，
`Q` 或 Esc 退出。默认应用真实 `plumb_bob` 畸变；使用 `--no-distortion`
可查看无畸变针孔图像。

将所有通过预处理的 replay 按固定 30 Hz 采集为本地 LeRobot v3 数据集：

```bash
uv pip install --python .venv/bin/python -r requirements-lerobot.txt
.venv/bin/python scripts/collect_a2d_lerobot.py
```

默认输出到 `collected_datasets/a2d_head_camera_roi_lerobot/`。相机先生成并应用畸变到
1280×800 图像，再按 `image[320:800, 126:974]` 保存 848×480 的
`observation.images.head_color` 视频，以及 16 维 `observation.state` 和 `action`。
16 维顺序为左臂 7 个关节、右臂 7 个关节、左右夹爪开度，夹爪仍使用
`0=闭合，1=张开`。每条样本满足 `action_t = state_{t+1}`；由于最后一个状态
没有下一帧动作，所以每个 episode 保存的 transition 数比采样状态数少 1。
采集脚本不会覆盖已存在的输出目录；试采一条数据可使用：

```bash
.venv/bin/python scripts/collect_a2d_lerobot.py \
  --output-dir collected_datasets/a2d_head_camera_test \
  --max-episodes 1 \
  --preview
```

使用 GR00T 策略服务器在同一 MuJoCo 场景中进行闭环推理验证：

```bash
uv pip install --python .venv/bin/python -r requirements-gr00t-client.txt
.venv/bin/python scripts/eval_mujoco_gr00t_genie1.py \
  --policy-host 172.20.103.219 \
  --policy-port 5555
```

默认加载第一个通过预处理的 episode 作为初始骰子和纸盒布局，并保持清单中的固定
躯干姿态。MuJoCo 主窗口和 848×480 策略输入窗口打开后，按空格开始/暂停，按
`Q` 或 Esc 退出。输入状态和输出动作均按
`[左臂7, 左夹爪, 右臂7, 右夹爪]` 排列。只检查本地场景、相机和观测形状而不连接
服务器时使用：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/eval_mujoco_gr00t_genie1.py \
  --dry-run --headless
```

保存 rollout 的策略输入视频和已执行动作：

```bash
.venv/bin/python scripts/eval_mujoco_gr00t_genie1.py \
  --policy-host 172.20.103.219 \
  --save-video logs/mujoco_gr00t_rollout.mp4 \
  --save-actions logs/mujoco_gr00t_actions.npz
```

窗口中按 `SPACE` 暂停或继续，按 `N`/`P` 切换下一条/上一条，按 `ENTER` 重新播放
当前 episode。需要连续自动播放全部通过的数据时使用：

```bash
.venv/bin/python scripts/replay_a2d_dataset.py \
  --speed 0.5 --start-immediately --auto-advance
```

需要人工检查报告中标记为失败、但仍生成了缓存和布局的数据时，可加
`--include-failed`。

播放器始终使用清单中的同一组 `body_lift` 和 `body_pitch`。每条轨迹只校正右臂：
先对齐桌面抓取高度，再在抓稳后按需要平滑抬高放置阶段；骰子和纸盒随后根据校正
后的抓取与释放位置自动摆放。

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
