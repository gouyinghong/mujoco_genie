# A2D 闭环夹爪物理回放

多条 episode 的自动准备、连续回放和结果汇总，见 [批量动力学回放](a2d_physics_batch.md)。

## 当前通过低滑移检查的命令

```bash
.venv/bin/python scripts/replay_a2d_physics.py \
  --manifest datasets/a2d_closed_loop_episode_000000_hold41_close2s/manifest.json \
  --episode episode_000000.npz \
  --grasp-lower-m 0 \
  --gripper-close-bias 0 \
  --gripper-sliding-friction 3 \
  --arm-contact-mode constrained \
  --physics-timestep 0.0005 \
  --contact-impratio 100 \
  --speed 0.5 \
  --dice-linear-damping 0.02 \
  --dice-angular-damping 0.0005
```

新增模式给 14 个手臂关节和 4 个固定躯干/头部关节增加给定运动约束。
此前每步覆盖手臂位置、速度，但接触求解仍允许这些关节受力加速；新模式使求解器
同时考虑它们应按给定轨迹运动。约束参考带速度补偿，避免约束阻尼把目标运动当作
需要刹停的速度。手指继续由原有限力矩执行器驱动；没有约束骰子，也没有增加手指刚性锁定。
该模式表达理想的轨迹支撑，不是有真实力矩上限的全身控制器，不能据此验证实机手臂负载能力。

这次组合使用 2000 Hz 物理步进、摩擦/法向阻抗比 100，关闭闭合偏置，直接使用数据中的
`0.35` 目标。手臂轨迹高度、骰子初始位置、原始数据、手指碰撞体、关节惯量和每手
1 N·m 驱动力矩上限均未调整。`--speed` 只控制显示速度。默认仍为 `legacy`，
不加新参数的旧命令会继续使用旧模型。

原位置及分别沿 x/y 偏移 ±1 mm、偏航 ±1° 的 7 组完整测试均通过原有低滑移判据
（相对平移 ≤2 mm、转角 ≤5°，持续双侧接触并最终落箱），没有放宽阈值。
这不是数学上的零滑动，也不代表任意轨迹都能成功。
结果与视频：`logs/a2d_constrained_grasp_validation/metrics.json`、`replay.mp4`、`stages.png`。

### 指尖形状和夹持姿态检查

STL 中存在法向约为 `(0.55, ±0.83, 0)` 的大平面，现有胶囊体并未沿这些面布置。
但在实际夹持姿态中，骰子位于这些平面的背侧，不能直接将最大平面认作有效指垫。
对照了倾斜薄盒、整指网格凸包，以及满足闭环的实验性平行联动，均未得到可替换的
低滑移方案，因此没有采用这些形状或联动改动。
`contact_audit.json` 记录旧模式与新模式在多个时刻的接触位置、施加给骰子的世界坐标系
力和绕骰子中心的力矩。试验记录在同目录的 `experiments/` 中。

临时直接修改惯量曾得到较小滑移，但重新计算模型常量/重新编译后未能保持结果，
因此该做法被排除。最终方案只使用编译时建立的运动约束，不修改惯量或依赖旧缓存。
速度补偿使用固定阻抗下 `b/k = 2 * timeconst * dampratio²`，参见
[MuJoCo 求解器参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。

## 旧配置的验证结论修正

> 以下历史配置的“成功”只说明持续夹持并落箱，不能说明没有滑动。
> 用户反馈后补测发现，降低 1.5 cm 的配置仍有约 29.8 mm 相对向下位移、
> 22.4° 相对转动；零降低且摩擦为 3 的配置约为 22.9 mm、54.3°。
> 这两套配置均未通过新增的低滑移检查，不能称为稳定不滑的抓取方案。

回放增加 `--headless` 后可查看 `max_dice_translation_in_gripper_m`、
`max_dice_rotation_in_gripper_deg`、`max_dice_drop_relative_to_gripper_m`。
参考姿态取闭合完成后的第一个物理采样，检查至插值开始张开前；
位置和转角在夹爪基座坐标系测量，向下位移则是相对固定在夹爪中的参考位置、
沿当前世界竖直方向的偏移。因此包含夹爪内部运动导致的物体移动，
不能仅凭这些数值区分接触滑动、滚动和手指继续闭合的贡献。

`low_slip_grasp_success` 在原持续夹持判据之外，要求最大相对平移 ≤2 mm、
最大相对转角 ≤5°；`low_slip_pick_and_place_success` 再要求最终落箱。
这是本项目明确选择的工程容差，不代表数学上的零滑动。
旧的 `success` / `pick_and_place_success` 保留原语义。

### 初始位置与闭合时间对照

`logs/a2d_pose_slip_search/coarse.json` 保存围绕当前布局 x/y 各 ±2 cm、
间隔 1 cm，以及偏航 ±30°、间隔 15° 的 125 组完整回放。
它们保持 `grasp-lower-m=0`、指尖摩擦 3、默认闭环参数和原盒子位置，
没有候选通过低滑移抓取落箱判据；这只排除了已测试的离散布局，不能证明所有位置都无解。
`durations.json` 记录闭合时间 3、4、6、8 秒的对照，也没有通过。
实验副本在 `/tmp`，没有修改原始数据或现用布局。

布局搜索脚本现在支持 `--objective low-slip`，此模式会检查后续落箱，
优先选择通过低滑移检查的布局，再比较成功搬运布局的相对平移和转角。
搜索中心仍为该脚本由轨迹计算的初始姿态，可能与回放读取的独立布局不同。
请用独立的 `--output` 保存实验结果；`best_effort` 不代表低滑移验证成功。

## 保持原轨迹高度（新增验证）

```bash
.venv/bin/python scripts/replay_a2d_physics.py \
  --manifest datasets/a2d_closed_loop_episode_000000_hold41_close2s/manifest.json \
  --episode episode_000000.npz \
  --grasp-lower-m 0 \
  --gripper-sliding-friction 3 \
  --speed 0.5 \
  --dice-linear-damping 0.02 \
  --dice-angular-damping 0.0005
```

沿用第 41 帧停留闭合 2 秒的数据，仅将指尖滑动摩擦系数从模型的 1.2 调为 3。
右手轨迹不降低，物体初始位置、碰撞体形状、1 N·m 驱动力矩上限和 500 Hz
物理步进均保持原配置；没有绑定或吸附骰子。省略新参数仍使用原来的摩擦值。
指尖接触优先级高于骰子，因此单独提高骰子的摩擦参数不能替代这个指尖参数。
这是仿真接触参数的调整，系数 3 尚未经实际材料测量标定。

`logs/a2d_zero_height_validation/metrics.json` 保存系数 3 和 4 的对照结果。
系数 3 在原位置、分别沿 x/y 偏移 ±1 mm、偏航 ±1° 的 7 组测试中均成功。
从闭合完成后 0.15 秒到开始张开前，所有采样步均有两侧承载接触，接触中断为 0，
最终骰子静止在盒内。这覆盖本条数据和这些小扰动，不代表任意轨迹都能成功。
视频和关键帧保存在同一目录的 `replay.mp4`、`stages.png`，没有覆盖此前降低高度的结果。

验证器新增 `retention_bilateral_contact_fraction`、`max_preopening_contact_loss_s`
和 `opening_start_s`。成功必须保持接触至实际张开起点；线性插值的张开起点是
首个增大开度样本的前一帧，本条为约 4.399 秒，不能只根据最后落箱判断抓取成功。

## 此前降低 1.5 cm 的已验证命令

```bash
.venv/bin/python scripts/replay_a2d_physics.py \
  --manifest datasets/a2d_closed_loop_episode_000000_hold41_close2s/manifest.json \
  --episode episode_000000.npz \
  --gripper-control closed-loop \
  --grasp-lower-m 0.015 \
  --speed 0.5 \
  --dice-linear-damping 0.02 \
  --dice-angular-damping 0.0005
```

空格开始/暂停，Enter 重置。增加 `--show-collision` 查看碰撞体，增加
`--headless` 执行完整回放并打印接触、闭环误差和落箱验证结果。

`--grasp-lower-m 0.015` 使用现有 IK 工具将右手轨迹向下平移 1.5 cm，
仅修改内存中的轨迹副本。该参数默认是 0；删掉它即可比较只延长闭合时间的效果。
机械臂仍按轨迹做运动学回放，手指及骰子通过动力学求解。

## 数据处理

原始 episode、原始关节缓存和 `datasets/replay_layouts.json` 均不修改。
新目录包含独立 episode、关节缓存、summary、manifest 和初始物体布局。
生成器拒绝使用已存在的输出目录：

```bash
.venv/bin/python scripts/prepare_a2d_grasp_hold.py \
  --episode episode_000000.npz \
  --hold-frame 41 \
  --close-duration-s 2 \
  --output-dir datasets/my_new_grasp_hold
```

处理将原闭合阶段的右手保持张开，到第 41 帧停止手臂，插入 60 帧逐渐闭合，
再继续原来的搬运轨迹。右手指令从 1 降至原数据的 0.35。帧数从 110 增至 170，
轨迹时长从约 3.63 秒增至 5.63 秒。左右手臂的原有姿态样本均保留；
插入帧重复停留姿态，后续时间戳整体延后 2 秒。

`parent_episode_frame_index`、`synthetic_hold_frame` 和 manifest 的 `processing`
字段记录映射、合成帧、源文件路径及 SHA-256。源测量/IK 信息按来源复制，
不表示对插入帧重新测量或重新运行了原始优化器。高度修正不写回这些文件。

`a2d_closed_loop_episode_000000_close2s` 是第 37 帧停留的对照版本；
推荐使用上面的第 41 帧版本。两个目录均独立于原数据。

## 夹爪实现

参考 [VLM_Grasp_Interactive 的 Robotiq 夹爪模型](https://github.com/hangtingLiu/VLM_Grasp_Interactive/blob/5cf719a7490d1a4993dd038e2804eaeadd51af23/manipulator_grasp/assets/robotiq_2f85/2f85.xml)
的闭环连接、双指同步和共享驱动结构，使用 A2D 自己的尺寸和关节方向：

- 4 号指尖连杆在局部 `y=±0.0105 m` 的第二销轴通过 `connect` 连接到 2 号连杆；
  对应连接点按模型原始零位计算。
- 原模型每指有 4 个转动关节，单独加几何闭环还存在额外自由度。
  3 号关节与主动 1 号关节之间使用现有 A2D 开合标定拟合的四次多项式联动约束。
  2、4 号关节由闭环几何关系求解。
- 同一只手的宽、窄主动关节反向同步。一个固定 tendon 将驱动力以 `+0.5/-0.5`
  分配到两侧；每只手只有一个位置执行器。
- 只有重置时初始化手指状态；播放和静置期间更新 `ctrl`，不重写手指 `qpos/qvel`。
- 闭环参数、执行器、接触设置通过运行时 `MjSpec` 构造，不改写原 XML。

这是一套包含实际几何闭环和标定联动的有效模型。联动拟合、惯量附加项和执行器
增益尚未经实机校准，不等同于完整重建了 A2D 的真实传动装置。

默认 `kp=30`、`kv=0.2`、每手驱动力矩上限 `1 N·m`，可用相应 `--gripper-*`
参数调整。`--gripper-close-bias 0.15` 使用
`clip(openness - 0.15*(1-openness), minimum, 1)` 作为目标；完全张开不变，
录制的 0.35 对应 0.2525，以提供接触后的夹持余量。实际开度由受力平衡决定。
增加 `--gripper-control kinematic` 恢复旧的手指位置回放方式。

## 验证

`tests/test_a2d_closed_loop.py` 检查开合往返的闭环连续性、双指同步、力矩限制、
控制命令不覆盖状态、数据不覆盖，以及这条 episode 的实际抓取和落箱。

慢闭合时两侧首次接触时间可能相差较大，原先要求“首次接触时差 ≤50 ms”的
启发式不适合作为抓取成功标准。它仍保留在 `legacy_layout_success` 中。
新的 `success` 要求：搬运区间至少 95% 的采样同时具有双侧有效接触力
（各自法向力 >1 mN），桌/箱支撑接触占比不超过 5%，抬升至少 8 cm，
夹爪距离 p90 不超过 6 cm，释放前位置误差不超过 8 cm，且没有数值警告、
过大速度、闭环脱离或机器人穿入支撑面。现在还要求闭合完成后 0.15 秒至开始张开
期间每一步均保持双侧有效接触，避免最后一段提前滑落。

`pick_and_place_success` 还要求后续落下稳定，并检查旋转骰子的完整包围范围
落在箱内，不能仅凭骰子中心位置判定。布局文件的旧状态与分数属于保存时的
元数据；实际运行结果以当前输出为准。

此前使用降低 1.5 cm 的命令，在 MuJoCo 3.9.0 下，标称位置以及 x/y 各 ±1 mm 的 5 次测试
均完成抓取和落箱。标称搬运阶段双侧有效接触占比为 100%，无桌/箱支撑，
骰子中心最高约 0.950 m（离初始中心约 12 cm），含释放后阶段的最大闭环误差约 0.043 mm，
没有机器人与桌/箱的穿透或物理警告。验证仅覆盖这条 episode 和这些小扰动。

结果：`logs/a2d_closed_loop_validation/metrics.json`；
渲染回放：`logs/a2d_closed_loop_validation/replay.mp4`（0.5 倍速）。
