# GR00T 动力学推理验证

脚本：`scripts/eval_a2d_physics_gr00t.py`。只读取测试示范的初始机器人姿态和场景，后续动作来自服务器。不会覆盖数据集，也不会播放示范的后续动作。

## 运行

有窗口，查看第 0 条测试场景：

```bash
.venv/bin/python scripts/eval_a2d_physics_gr00t.py \
  --policy-host 172.20.103.219 --policy-port 5555 \
  --episode-index 0 --preview --realtime --save-video
```

默认使用 `datasets/a2d_augmented_v2/collection/test/lerobot`。可用 `--dataset` 指定其他采用相同采集物理参数的数据目录。除了 LeRobot 目录，还需要其上一级的 `collection_report.json` 及报告引用的 prepared manifest、模型和轨迹文件；这些文件用于恢复完整环境，不能只复制 parquet/video。

当前测试集索引与扩增场景对应：

| `--episode-index` | 扩增场景 |
| --- | --- |
| 0 | episode_000083.npz |
| 1 | episode_000104.npz |
| 2 | episode_000125.npz |
| 3 | episode_000146.npz |
| 4 | episode_000167.npz |
| 5 | episode_000188.npz |

无窗口批量验证全部 6 个场景：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/eval_a2d_physics_gr00t.py \
  --policy-host 172.20.103.219 --policy-port 5555 \
  --all-episodes --headless --save-video
```

仅检查场景、状态、相机，不连接服务器：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/eval_a2d_physics_gr00t.py \
  --all-episodes --headless --dry-run
```

默认每个场景最多 600 个控制步，即约 20 秒仿真时间；使用 `--max-steps` 调整。每次推理默认执行动作块的前 8 步后重新观察，使用 `--replan-steps` 调整，0 表示执行整个动作块。每个场景开始调用服务器 reset。网络推理期间暂停物理时间，因此该评估不包含真实控制延迟的影响。

`--start-paused` 可让窗口初始暂停，空格切换暂停。关闭仿真窗口或相机窗口按 q/ESC 中止当前评估。

## 数据与物理一致性

- 观测为实际关节位置和实际夹爪开度，RGB 为与采集一致的 848×480 头部相机裁剪，恢复场景的光照、桌面颜色和盒子纹理参数。
- 数据集 state/action 顺序：左臂 `[0:7]`、右臂 `[7:14]`、左夹爪 `[14:15]`、右夹爪 `[15:16]`。这是绝对位置目标，夹爪开度 0 为闭合，1 为打开。
- 请求使用参考脚本的命名模态 `video.ego_view`、`state.left_arm/right_arm/left_gripper/right_gripper`、`language.task_description`。服务器的训练数据切片配置必须与上述数据集顺序一致。
- 支持服务器返回命名动作模态，或 `(16,)` / `(H,16)` / `(1,H,16)` 扁平 action。扁平返回默认同数据集顺序；只有服务器实际返回左右臂各自紧邻夹爪的顺序时，才设置 `--flat-action-order interleaved`。
- 控制频率读取数据集（当前 30 Hz），物理步长 0.0005 秒（2000 Hz）。手臂目标逐物理步插值，并使用现有接触约束；手指由有限力矩执行器驱动，最大 1 Nm，摩擦 3，骰子保留原阻尼。
- 手臂沿用采集时的规定运动约束，并非完整的有限力矩手臂控制。夹爪和骰子通过动力学与接触运动。
- 动作执行限制关节范围、夹爪开度和手臂目标速度（默认 2 rad/s，`--max-joint-speed` 可调）；报告记录被限制的动作数，原始预测和实际使用目标均保留。
- 训练数据中的 fast 放开已经写入动作标签，推理不额外对开度增量触发强制张开，也不自动补上 2 秒闭合；这些时序由模型预测。

## 结果

每次自动新建 `logs/a2d_gr00t_physics_<时间>/`，也可用 `--output-dir` 指定一个不存在的目录。

- `summary.json`：参数、场景逐条结果。
- `episode_XXXXXX/initial.png`：初始模型输入画面。
- `result.json`：状态、是否成功、推理次数、动作限制次数和物理指标。
- `rollout.npz`：`state`、`proposed_action`、实际使用的目标 `action`、`next_state`、`next_dice_qpos`。`state_time_s` 对应动作前，`next_time_s` 对应动作后；时间从初始化静置完成后计。
- `rollout.mp4`：指定 `--save-video` 时生成，画面对应每步动作前的状态。

成功要求曾有双侧指垫持续接触至少 0.1 秒，并在接触中抬高骰子至少 8 cm；之后整个骰子落在盒内、线速度低于 1 cm/s、无指垫承载接触，连续保持默认 10 个控制步。同时要求无物理警告、闭环误差低于 1 mm、机器人与桌面/盒子的最大穿透不超过 1 mm。这是自动指标，可结合视频判断动作质量。

`timeout` 表示达到步数上限但未满足上述成功标准，`unstable` 表示物理警告/非有限状态，`error` 表示接口或执行错误。接口错误会停止后续场景，避免将断线误算成整批模型失败。全部成功（或全部 dry-run）退出码为 0，否则为 2。

本地测试：

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_eval_a2d_physics_gr00t.py -q
```
