# A2D 批量动力学回放

在项目根目录运行。新入口默认使用 `datasets/replay_layouts.json` 的总清单，
而不是此前只包含 `episode_000000.npz` 的单条清单。

## 连续可视回放

```bash
.venv/bin/python scripts/replay_a2d_physics_batch.py
```

逐条准备数据、打开回放窗口并自动开始；该条完成后窗口自动关闭，再进入下一条。
每条使用独立进程，避免上一条的仿真状态影响下一条。SPACE 暂停/继续，ENTER 重置当前条。
主动关闭尚未完成的窗口或按 Ctrl+C 会中断整批，不会直接跳到下一条。

批量入口默认包含已经验证的配置：`closed-loop`、`grasp-lower-m=0`、
`gripper-close-bias=0`、指尖摩擦 `3`、`arm-contact-mode=constrained`、
物理步长 `0.0005`、`contact-impratio=100`、显示速度 `0.5`、骰子阻尼 `0.02/0.0005`。
可以用对应命令行参数覆盖。单条回放脚本原有默认值没有更改。

可视模式完成后，会从重置状态再运行一次无窗口验证以保存指标，
因此报告不是对鼠标拖动、施加外力等人工干预过程的测量。

## 无窗口批量验证

```bash
.venv/bin/python scripts/replay_a2d_physics_batch.py \
  --headless \
  --output-dir logs/my_physics_batch
```

默认输出目录是 `logs/a2d_physics_batch/时间戳/`。手动指定的目录必须尚不存在，
除非使用 `--resume`。不限制进程运行时间；可用 `--timeout-s 120` 设置每条进程的
墙钟超时。可视模式下暂停也计入超时，通常应保留默认 `0`（不限时）。

选择部分数据：

```bash
# 前 5 条清单记录
.venv/bin/python scripts/replay_a2d_physics_batch.py --headless --limit 5

# 指定文件，按清单顺序执行
.venv/bin/python scripts/replay_a2d_physics_batch.py --headless \
  --episodes episode_000001.npz episode_000002.npz
```

`--start-index` 与 `--limit` 作用于筛选后的清单顺序，包含需要跳过的原失败记录。
输入仍需是带有逐条布局、固定躯干和关节缓存信息的 replay manifest，不能仅提供 NPZ 目录。

## 数据处理与输出

默认对每条 `status=ok` 的原始记录调用现有闭合处理器：

- 使用该条自己的 `close_end_frame` 停留，不固定使用第 41 帧。
- 在停留位置逐渐闭合 2 秒；开度端点取自各条数据，不强制所有条目为 `0.35`。
- 复制、扩展关节缓存并平移后续时间戳，保留手臂原有姿态样本和该条骰子/盒子布局。
- 每条生成独立的 `data/episode_xxxxxx/manifest.json` 和数据副本，不覆盖原始文件。

用 `--close-duration-s` 调整闭合时长。`--keep-timing` 可直接回放现有时序；
如果传入的是已有 `stationary_gripper_closure` 标记的处理后清单，会自动避免再次插入闭合帧。

输出目录包含：

| 路径 | 内容 |
|---|---|
| `configuration.json` | 参数和主要源文件 SHA-256，用于续跑一致性检查 |
| `data/` | 每条独立的闭合处理副本 |
| `results/episode_xxxxxx.json` | 完整动力学验证指标 |
| `logs/episode_xxxxxx.log` | 子进程输出与错误日志 |
| `summary.json`、`summary.csv` | 逐条状态、关键指标、路径及汇总计数 |

`passed` 表示通过原有低滑移抓取落箱判据；`failed` 表示仿真执行完毕但未通过判据；
`error` 表示数据准备、进程或指标读取错误；`skipped` 表示原清单已有失败状态。
每条结束都会保存报告，失败不会阻止后续条目执行。

## 中断后继续

保持原命令参数并增加 `--resume`：

```bash
.venv/bin/python scripts/replay_a2d_physics_batch.py \
  --headless --output-dir logs/my_physics_batch --resume
```

已完成的 `passed`、`failed`、`skipped` 不重复执行；未完成、被中断、执行错误的记录继续尝试。
参数或主要输入文件变化会拒绝续跑，需要使用新目录。修改参数重试物理抓取失败的条目时，
请用 `--episodes` 和新输出目录。部分写入的准备目录会保留，重试另建目录。

退出码：`0` 为无失败/执行错误，`1` 为存在物理判据失败或执行错误，`130` 为用户中断。
退出码 `1` 不表示批量过程提前停止，应以 `summary.json` 为准。

## 放置时夹爪迟迟不松开

可只对指定 episode 启用快速张开目标：

```bash
.venv/bin/python scripts/replay_a2d_physics_batch.py \
  --episodes episode_000006.npz episode_000010.npz episode_000020.npz \
  --gripper-release-mode fast
```

默认 `recorded` 沿用录制的开度变化。`fast` 在某只手的原始开度指令开始增大时，
将该手执行器目标设为全开，并在随后的平台段保持；收到新的闭合指令或重置时取消。
闭合和搬运目标不变，仍保留每手 1 N·m 力矩上限。只修改运行时控制目标，
不会瞬移手指、禁用碰撞、改变摩擦或覆盖数据。此选项适用于当前清单的明确开合指令；
若指令带有开度噪声，需要先识别真实松手阶段，不能把任意微小上升都当作松手。

6、10、20 的驱动记录显示，开始增加开度时，目标仍小于骰子撑开的实际驱动位置，
原控制器还施加 -1 N·m 闭合力矩；快速模式此时转为 +1 N·m 张开力矩。
在 2000 Hz 下，最后有效指尖接触相对张开起点的延迟分别从约 137、75、141 ms
降至约 5、5、5 ms，三条最终都落箱。保持原控制而提高到 4000/8000 Hz，延迟基本不变。
这里测量的是最后有效接触时刻，可能包含再次碰触，不表示检测到了材料粘附力。

新增报告字段 `postopening_finger_contact_duration_s` 是张开后仍存在有效指尖接触的累计时长，
`last_finger_contact_delay_from_opening_s` 是最后一次有效接触相对张开起点的延迟。
有效接触阈值沿用法向力 >1 mN。

结果保存在 `logs/a2d_fast_release_validation/`，包括频率对照、驱动力矩曲线数据和第 6 条视频。
10、20 的搬运位移仍略超旧的 2 mm 阈值，所以整体报告仍可能标为 `failed`；
快速松手不改变此前搬运结果，也不放宽原判据。用户认为可接受的 8、12、14、28
未在此次更改默认行为；原始数据与旧的逐条指标文件未改动。

## 本次验证

当前总清单的 31 条记录已完整跑过：29 条执行，20 条通过低滑移检查、9 条未通过，
2 条按原失败标记跳过，0 条执行错误。不同轨迹仍需分别验证，批量功能不保证每条都能成功抓取。
结果位于 `logs/a2d_physics_batch_validation/summary.csv`。
已核对所有原始 episode 和关节缓存的 SHA-256，均未改变；续跑会跳过本次已完成记录。

测试覆盖逐条失败隔离、续跑、输入变更拒绝、选择范围，以及用模拟窗口验证自动关闭与用户中断。
无窗口流程已实际运行；原生桌面窗口的显示交互仍需在有显示环境的机器上运行上面的可视命令检查。
