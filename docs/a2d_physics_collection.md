# A2D 动力学 LeRobot 采集

独立入口 `scripts/collect_a2d_physics_lerobot.py`，不修改 `collect_a2d_lerobot.py`。
输入是当前 `datasets/replay_layouts.json` 和对应原始数据/关节缓存。
如果替换了原始 episode，请先重新预处理对应记录。

```bash
.venv/bin/python scripts/collect_a2d_physics_lerobot.py
```

默认排除 `episode_000003.npz`、`episode_000013.npz`、`episode_000015.npz`、
`episode_000023.npz`，即使通过 `--episodes` 指定，也不会采集这些数据。
同时跳过 manifest 中状态不是 `ok` 的记录。当前清单共 27 条符合条件。
不会删除原数据，不按 2 mm 滑动阈值额外过滤；每条的验证结果保存在报告中。

默认 2 秒闭合、fast 松手、30 FPS、轨迹结束后额外录制 1 秒。
每条在新目录生成闭合数据副本。只接受未经闭合处理的总清单，避免重复插入停顿。

```bash
# 使用已对照测试过的 1.5 秒闭合
.venv/bin/python scripts/collect_a2d_physics_lerobot.py --close-duration-s 1.5

# 先采集指定数据并预览（Q/Esc 中止，未完成的一条不保存）
.venv/bin/python scripts/collect_a2d_physics_lerobot.py \
  --episodes episode_000000.npz episode_000008.npz --preview

# 无桌面环境使用 EGL
MUJOCO_GL=egl .venv/bin/python scripts/collect_a2d_physics_lerobot.py \
  --output-dir collected_datasets/my_physics_collection
```

默认输出为 `collected_datasets/a2d_physics_<时间戳>/`。指定已有目录会报错，不能覆盖或续写。
输出内容：

- `lerobot/`：真正的 LeRobot 数据集根目录，包含视频、Parquet 和 meta；训练时使用此子目录。
- `prepared/`：每条闭合时间处理后的 NPZ、关节缓存、布局和来源信息。
- `collection_report.json`：参数、排除名单、验证指标，以及连续 LeRobot 索引到原始 episode 的映射。
- `episode_XXXXXX_physics.npz`：按 LeRobot 索引保存的物理状态记录，用于检查图像与状态的对应。

仿真使用与已验证 batch 相同的参数：0.5 ms 物理步长、constrained 手臂轨迹约束、
impratio 100、夹爪摩擦 3、kp 30、kv 0.2、最大驱动力矩 1 N·m、close bias 0、
手臂下移 0、骰子线性阻尼 0.02/角阻尼 0.0005、初始化静置 0.4 秒。
骰子只在初始化设置位置，之后完全由物理仿真运动；没有绑定夹爪或预设掉落轨迹。
手臂仍是已有的轨迹约束回放，不代表整个机器人都由有限力矩执行器控制。

数据字段与旧采集兼容，但状态/动作语义有意区分：

- `observation.images.head_color`：头部相机 RGB，经原相机畸变处理，裁剪为 848×480。
- `observation.state`：14 个实际手臂关节角（弧度）与左右实际夹爪开度。
  开度按驱动关节的差值归一化到 [0,1]；细微越界裁剪，原始值保存在物理记录中。
- `action`：下一采样时刻的有效位置控制目标，14 个手臂目标角加左右归一化夹爪驱动目标。
  fast 松手使用实际全开目标，不使用原始数据中缓慢增加的开度，也不使用下一帧实测位置作为目标。

这里沿用旧脚本的“当前图像/状态 → 下一采样时刻目标”约定。
控制目标在 2000 Hz 更新，30 FPS 数据不是完整的高频控制日志，训练后仅在 30 Hz
应用动作不保证复现完全相同的接触过程。LeRobot 时间戳均匀，采样实际量化到最近
0.5 ms 步长（误差不超过 0.25 ms），准确仿真时刻另存 `sample_times_s`。
物理 NPZ 的状态/时刻/骰子姿态有 N+1 个采样，action 和视频有 N 条，最后一个采样
只用于生成上一帧的动作标签。图像渲染在 MjData 副本上进行，不改变运行中的求解器状态。

每条在渲染前另做一次同参数无图像验证，`validation_metrics` 来自该独立回放；
实际采集轨迹另存骰子位置和姿态 `dice_qpos`。报告不将这些指标冒充为采集视频的逐帧测量。
脚本不自动上传数据。缺少依赖时会提示使用已有 `requirements-lerobot.txt` 安装。
