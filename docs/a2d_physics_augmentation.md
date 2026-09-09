# A2D 物体参照轨迹扩增

入口 `scripts/generate_a2d_physics_augment.py`，核心算法在 `scripts/a2d_augmentation.py`。
借鉴 MimicGen 的物体参照子任务变换方法，在本项目的 MuJoCo 环境独立实现；
没有引入 robosuite，也不是调用官方 MimicGen。
参考：https://mimicgen.github.io/docs/tutorials/datagen_custom.html

## 使用

```bash
# 最多尝试 200 条候选，取得 100 条合格轨迹后提前结束
.venv/bin/python scripts/generate_a2d_physics_augment.py \
  --attempts 200 --target-successes 100 \
  --output-dir datasets/my_a2d_augmentation

# 合格轨迹直接采集为独立的 train/test LeRobot 数据集，同时改变外观
MUJOCO_GL=egl .venv/bin/python scripts/generate_a2d_physics_augment.py \
  --attempts 200 --target-successes 100 \
  --visual-randomization --collect \
  --output-dir datasets/my_a2d_augmentation_with_video
```

默认使用 `datasets/replay_layouts.json`，2 秒闭合、fast 松手。
每条原始数据先验证，选取前 5 条严格通过的示范作为模板；可以用 `--episodes`
指定候选来源、用 `--max-sources` 修改模板数量。始终排除原始 3、13、15、23。
`--close-duration-s 1.5` 会重新验证来源，不会假定较短闭合时间一定成功。
原始数据、原关节缓存和已有报告都不修改。已有输出目录会被拒绝，不能续写。

默认采样范围（相对于来源场景）：

| 参数 | 默认值 |
| --- | --- |
| `--dice-xy-range-m` | X/Y 各 ±0.02 m |
| `--box-xy-range-m` | X/Y 各 ±0.03 m |
| `--dice-yaw-range-deg` | ±10° |
| `--box-yaw-range-deg` | ±5° |
| `--seed` | 2026 |

一次尝试不保证成功，`--target-successes` 是目标而非保证。
达到 `--attempts` 上限时保留所有已通过的数据，并在报告中记录 `target_reached=false`。
同样种子与相同参数、源码和输入可复现采样；改变外观随机化开关也会改变随机数序列。

## 流程

1. 为来源生成独立的闭合时间副本，并用当前动力学参数验证。
2. 从固定躯干下的关节轨迹通过正运动学提取右夹爪基座世界位姿。
3. 接近和闭合段相对骰子变换；搬运段平滑转向盒子参照；放置段相对盒子变换。
   松手后的撤回段平滑接回来源的最终休息姿态。
4. 六维阻尼最小二乘逆运动学同时追踪位置/方向，沿用上一帧关节修正保持肘部解连续。
   超出误差、关节限位或单帧步进门槛的候选直接拒绝。
5. 按最大关节速度 2 rad/s 延长必要的运动区间；闭合区间时长保持不变。
   这会改变新数据运动段的时间戳，不只是更换物体位置。
6. 用真正的动力学试跑到闭合后 0.15 秒，测量实际骰子相对夹爪的位姿。
   与来源的实际夹持关系比较，修正放置段目标，再做一次 IK。
   此步骤不绑定骰子、不清零骰子速度，也不修改接触参数。
7. 检查桌面碰撞，保存候选文件，再从文件重新加载并运行完整动力学验证。
8. 合格候选进入总清单及 train/test 清单；失败原因和已生成的失败候选另存，便于重放。
9. 可选采集 30 FPS 头部相机 RGB、实际状态、下一采样时刻控制目标。

阶段边界使用来源开合边界，结合闭合后实际动力学夹持测量。
第一版在一条来源示范内部变换各阶段，不混拼不同来源的夹持片段。
当前只支持右手抓放。手臂沿用 constrained 轨迹支持，夹爪有限力矩和骰子动态接触；
不是完整机器人的有限力矩控制仿真。

## 质量门槛

来源模板要求原来的严格通过标准。扩增输出允许有限的轻微滑动，明确使用独立标准：

- 骰子完整落入盒内并稳定，最大携带高度至少 0.91 m（当前桌面 0.8 m、骰子中心初始 0.83 m）。
- 搬运双侧受力接触比例至少 95%，保留阶段至少 98%，连续接触丢失不超过 20 ms。
- 相对滑动不超过 10 mm，旋转不超过 10°；可通过 `--max-slip-m`、`--max-rotation-deg` 修改。
- 搬运期间桌面/盒子支撑比例不超过 5%，机器人与支撑物穿透不超过 1 mm。
- 无仿真警告，夹爪连杆误差小于 1 mm，骰子速度不超过 2 m/s。

这不是修改原 batch 的 2 mm/5° 标准。扩增数据可能通过本流程，但在旧 batch 中仍标为
`failed`，需要根据报告中的滑动和接触指标判断。碰巧落箱而未正常保持夹持的数据会被拒绝。

## 输出与检查

- `manifest.json`：所有合格的新数据，兼容动力学 batch 和新采集脚本。
- `train_manifest.json` / `test_manifest.json`：训练/测试划分。
- `data/`：合格 NPZ 和 summary。
- `sources/`：来源闭合副本。
- `candidates/`：已生成候选的 NPZ、修正缓存、独立 manifest 和物理布局。
  在 IK 阶段就被拒绝的候选只有报告参数和原因，没有有效轨迹文件。
- `generation_report.json`：来源哈希、参数、每次尝试的变换、判定、指标和清单路径。
- 使用 `--collect` 后：`collection/train/lerobot/` 和 `collection/test/lerobot/`，分别为训练和测试数据集根目录。

生成编号是尝试编号，会有空缺。新 `episode_000003.npz` 不代表原始第 3 条；
来源保存在 `source_episode` / `augmentation.source_episode`。排除规则按原始来源执行。

```bash
# 查看某条生成轨迹，不能给新数据再插入一次闭合停顿
.venv/bin/python scripts/replay_a2d_physics_batch.py \
  --manifest datasets/my_a2d_augmentation/manifest.json \
  --episodes episode_000000.npz --gripper-release-mode fast

# 单独采集训练部分；自动保持扩增清单的闭合时间
MUJOCO_GL=egl .venv/bin/python scripts/collect_a2d_physics_lerobot.py \
  --manifest datasets/my_a2d_augmentation/train_manifest.json \
  --output-dir collected_datasets/my_augmented_train
```

`--visual-randomization` 会为每条保存灯光强度、桌面颜色和纸箱纹理 gamma 参数。
它们仅在采集时应用，不改变碰撞几何、质量、摩擦或相机内参；普通 batch 的 GUI
目前只复现动作和布局，不应用这些外观变化。骰子尺寸、质量、摩擦和动作噪声暂不随机化，
先单独验证位置/朝向变化，避免混淆接触问题来源。

## 泛化评估边界

用骰子和盒子的绝对 X/Y 所在 1 cm 网格及 5° 朝向区间确定场景组，稳定哈希划分
约 80% train / 20% test。在至少两个来源时，最后一个来源专用于 test，其他来源专用于
train；采样时同时满足场景组划分，因此训练/测试既没有相同场景组，也没有相同来源。
只有一个来源的试运行只能做到场景组隔离，不能做到来源隔离。

这验证的是未见过的局部布局组合，不等于更大范围的空间外推。模型训练、跨区域策略
评估仍需后续开展，生成成功率也不等于策略成功率。
如果把旧示范加入训练集，应排除这里留给 test 的来源示范，不能无条件把原始 27 条都
混进训练，否则会破坏来源隔离。要评估更大范围，可用更大的扰动范围和不同来源另生成测试集。

汇总通过率、失败原因、来源比例和实际扰动覆盖范围，并检查训练/测试隔离：

```bash
.venv/bin/python scripts/summarize_a2d_augmentation.py \
  datasets/my_a2d_augmentation/generation_report.json \
  --output-prefix datasets/my_a2d_augmentation/summary
```

该命令生成 `summary.csv` 和 `summary.json`，已有同名汇总时拒绝覆盖。
生成过程中也可以省略 `--output-prefix`，只查看当前进度。

## 已完成的首批数据

`datasets/a2d_augmented_v1/` 已完成生成、采集和完整性检查：

- 175 次候选尝试，保留 100 条；75 条训练数据、25 条测试数据。
- 训练集根目录：`datasets/a2d_augmented_v1/collection/train/lerobot/`，14,362 帧。
- 测试集根目录：`datasets/a2d_augmented_v1/collection/test/lerobot/`，4,488 帧。
- 全部视频逐帧解码通过，视频数量与 Parquet 状态/动作记录一致；精确核对了物理 NPZ 与每条状态/动作。
- 实际采集最终位置与独立验证的位置差异最大约 0.041 mm。
- 来源及场景组隔离通过；这批测试数据来自原始 `episode_000005.npz`，混入原数据训练时需要保留这个来源作为测试用途。
- 全部轨迹保持 2 秒静止闭合，运动关节速度不超过约 2 rad/s；原始模型、清单、输入 NPZ 和关节缓存哈希不变。

证据文件：`summary.json`、`trajectory_audit.json`、`collection_audit.json` 和各 split 的
`collection_report.json`。`coverage.png` 展示布局覆盖，`visual_stages.jpg` 展示实际采集帧。
27 项相关测试通过；另对试验批次的 25 条保留轨迹使用原 batch 回放，25 条均通过严格标准。
这些是生成及采集验证，尚未进行模型训练后的策略泛化对照实验。
