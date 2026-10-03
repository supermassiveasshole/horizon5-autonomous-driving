# 有界 SAC 采样与学习循环（#11）

`run_experiment(SACCycle(...), sac_environment=...)` 已支持响应策略动作的合成外部环境：固定完整策略采样、独立结算、追加经验、继续同一 learner、重载候选，再开始下一次尝试。实际执行 CPU Torch 更新；这是自动循环的软件证据，不是 FH5 物理模拟、游戏驾驶或能力改善证明。原生游戏采样、共享 GPU 时效与驾驶对照仍待验收。

## 实验入口和适配器

```python
from pathlib import Path
from fh5.experiment import run_experiment
from fh5.sac_cycle import SACCycle

result = run_experiment(
    SACCycle(
        checkpoint_dir=Path("runs/sac-initial"),
        recording_config_file=Path("configs/policy-record.json"),
        task_file=Path("runs/task.json"),
        reward_file=Path("configs/reward.json"),
        output_dir=Path("runs/sac-cycle"),
        cycles=3,
        steps_per_attempt=16,
        seed=19,
    ),
    sac_environment=environment,
)
```

路径须指向已有且一致的实验资产；`environment` 是调用方提供的 `SACEnvironment`，当前必须声明 `source_kind="synthetic"`。软件测试适配器见 `tests/test_sac_cycle.py`。不把任意实机适配器改一个来源标签后用于此入口。

- `start(epoch, pixels)` 返回新的 `SACStart`：新尝试的 `SACSample`、已确认执行的中立命令及其时刻；适配器负责重置自己的合成状态并准备属于新 epoch 的图像历史。
- `step(command)` 同步应用命令，返回随后真实产生的合成遥测及数值观测；程序使用模型实际输出，环境必须响应该命令。成功返回表示这一合成接口已应用输入，不能将这个同步时间口径搬到原生发送函数。
- `finish(recording_dir)` 结束尝试并返回独立任务证据文件，或返回 `None`；没有证据的片段不能给 learner 提供有效奖励。策略自身预测不能代替独立判定。
- `close()` 释放外部资源并报告结果。调用异常保留故障记录，停止后续尝试和学习。

每次尝试使用完整冻结的 encoder/actor，learner 在另一组模块中更新；续训检查实际读取的父清单摘要必须等于采样快照，目录被换成另一合法模型也拒绝接续。只在尝试结束、候选保存并重载核验后切换下一轮的探索版本。默认可靠版本不变，正式晋升仍由 #14 决定。采样 RNG 与 learner RNG 分离；每次决策保留噪声、上下文、实际整数命令及快照摘要。因果动作历史只由本次尝试严格早于当前决策的成功命令构造，失败或未知下发不继承为执行动作。

推理直接读不可变内存 RGB，并使用实际帧时间特征；采样完成后再归档数值字节。此同步测试适配器不承担原生实时输入租期、异步采集或硬截止保障，不能替代 #36/#9 的运行管线。

## 异步运行器中的探索策略

`SACSamplingActor(checkpoint, pixels, expected_sha256, exploration_seed=37)` 可作为
`run_experiment(RealtimeRun(...), realtime_environment=..., numeric_actor_factory=...)`
的冻结策略工厂。当前仅用于 `synthetic` 外部适配器，使用实际 CPU SAC；没有开放原生 SAC
驾驶或新的游戏输入权限。确定性评估仍使用 `SACEvaluationActor`，原有记录格式不变。

探索复用已训练策略的高斯分布和动作区间，不用全范围均匀随机起步。区间依据上条成功发送的整数命令、
当前决策距该命令返回的真实时间及检查点内的幅度/变化率限制形成。独立运行器继续决定是否接纳预测、
跳过决策、归零或停止，记录提议、实际发送、控制归属和时间；发送返回只是应用时刻的代理，不能证明游戏何时采用命令。

噪声采用版本化的 `sha256-box-muller-decision-v1`：无符号 64 位种子、epoch、决策编号和决策时间共同确定
两个标准正态分量。它不消耗跨决策的 RNG 状态；预热、跳帧或只回放部分已记录预测不会使后续噪声错位。
清单保留算法、种子和 CPU 设备，同一记录可经 `RealtimeNumericReplay` 独立重放。
不同运行的真实决策时间不同，因此相同种子不保证两次在线运行发出相同命令。

运行器保存异步输入、预测和命令证据，随后可用下面的实验入口准备学习经验。
原生资格校验、CUDA 推理及实际驾驶收益仍待后续接入和验证。

## 异步记录进入学习

`SACRealtimePrepare` 接收封存的异步执行目录、同一次尝试的完整遥测、独立任务/奖励与核验依据，
以及当时实际使用的冻结 actor。先核对原始包流、实际成功发送、因果动作历史和独立数值重放，
再生成 `sac-numeric-replay-v3`，可交给已有 `SACResume` 继续更新。
冻结 temporal BC 的记录也可准备经验，再经 `SACCriticWarmup` 与 `SACTrain` 启动 SAC，
无需先有 SAC 策略。BC 的前序动作与时间取自成功发送账本；首条缺少前序命令时排除，不补中立动作。

合成来源仍须验证发送返回至下一观测间的踏板/转向反馈；反馈未知或矛盾的片段不能进入学习。
原生来源仅接纳通过现有条件与影子资格的 `numeric-drive` 冻结 BC 记录，要求完整命令/UDP 绑定、
一致的独立任务路线与有效性依据，以及发送返回后、后继发送前的新遥测。
动作标签是实际成功发送的整数命令，不把 FH5 的滤波后 `Steer` 当作 XInput。
这些证据证明发送与随后观测，不能证明游戏采用命令的时刻或具体物理响应；
产物保留 `game_application="unverified"` 和发送返回代理时间，`real_driving_validated` 仍为 `false`。
经验来源沿 replay、critic 预热与 SAC 检查点传递；合成/原生混合标记为 `mixed`，不由父 BC 来源推断。
外部设备替身验证的是软件路径，不是真实游戏证据；本入口不开放原生 SAC 驾驶或自动实机采样。
准备前固定执行清单摘要，封存前复核未变，避免给旧转移绑定后来替换的来源。

```python
from fh5.sac_realtime_experience import SACRealtimePrepare
from fh5.sac_learning import SACResume

prepared = run_experiment(
    SACRealtimePrepare(
        recording_dir=Path("runs/attempt/recording"),
        execution_dir=Path("runs/attempt/execution"),
        task_file=Path("runs/task.json"),
        reward_file=Path("configs/reward.json"),
        output_dir=Path("runs/attempt/experience"),
        evidence_file=Path("runs/attempt/evidence.json"),
    ),
    numeric_actor=frozen_sampling_actor,
)
run_experiment(
    SACResume(
        checkpoint_dir=Path("runs/sac-initial"),
        output_dir=Path("runs/sac-next"),
        steps=2,
        additions=(
            (
                Path("runs/attempt/experience/replay.json"),
                prepared.summary["sac_replay"]["replay_sha256"],
            ),
        ),
    )
)
```

`frozen_sampling_actor` 必须与记录的模型、探索种子及时间特征契约一致。准备操作不会发送输入，
也不自动训练；示例假定已有至少两条可接纳的新经验和兼容的父检查点。

三种时间分别保留：

- `physical_dt_s`：当前观测到下一观测所对应的遥测游戏时间；奖励和折扣沿用独立逐包结算。
- `hold_dt_s`：本次成功发送返回至下一条命令发送返回的完整持有时间。
- `next_action_elapsed_s`：下一次**决策时**距本次成功发送返回的时间，用于下一状态的动作区间和 Q 目标。

不能用完整 hold 代替下一决策时已知的时间：后续推理和发送的延迟在下一决策时尚未发生。
观测锚点也不伪装成发送完成瞬间的游戏状态；从决策到发送返回期间，上一命令仍在生效。
记录保留上一命令、决策/发送时间和实际整数动作，发送返回仍只是应用时刻的代理。

缺一份图像存档只排除依赖它的转移，不补相邻帧；监督器归零不作为策略动作，之后完整策略片段仍可使用。
恢复前后的 epoch 和历史不能相连。独立确认的任务失败保留负回报和终止；没有真实下一观测的尾部排除，
不把录制结束伪造为零价值终止。独立核验未覆盖时没有合格经验。
目前完整命令账本必须全部成功发送且记录最终释放；发送失败或账本损坏会拒绝整份准备，尚不抢救其前缀。
旧同步 replay 与 v3 可按兼容性检查混合，各自时间依据保留，不改写成同一来源。

验证与已知边界见[异步经验接入记录](validation/t10-asynchronous-experience.md)。

## 自动交替异步采样与学习

`SACRealtimeCycle` 将上述异步运行、独立经验审核和续训接成有界循环。
它复用 `RealtimeRun` 的采集、推理、动作租期及旁路存档；学习只在采样资源全部释放之后进行。
当前使用实际 CPU SAC 和合成外部 I/O；[连续学习循环](learning-loop.md)版本 3 已接入异步采样、
冻结评估、保存与封存结果恢复。原生 SAC 资格仍未接入。

```python
from fh5.realtime import RealtimeConfig
from fh5.sac_cycle import SACRealtimeCycle

result = run_experiment(
    SACRealtimeCycle(
        checkpoint_dir=Path("runs/sac-initial"),
        recording_config_file=Path("configs/policy-record.json"),
        task_file=Path("runs/task.json"),
        reward_file=Path("configs/reward.json"),
        output_dir=Path("runs/sac-async-cycle"),
        runtime=RealtimeConfig(pixels=trained_pixel_contract),
        seconds_per_attempt=10,
        cycles=2,
        max_updates_per_attempt=8,
        expected_checkpoint_sha256=initial_manifest_sha256,
    ),
    sac_realtime_environment=environment,
)
```

示例中的像素、历史长度、参考槽位及动作幅度必须与实际检查点一致；不一致时在取得输入设备前拒绝。
`SACRealtimeEnvironment.start(identity, runtime)` 返回一次新的 `RealtimeEnvironment`；
它须准备新的观测/控制历史并对命令产生合成反馈。`finish(recording_dir)` 在该次运行器资源全部释放后
返回独立核验文件或 `None`，`close()` 释放环境级资源。适配器示例见 `tests/test_sac_realtime_cycle.py`。

每次执行保存完整命令/数值图像记录及原始遥测，封存来源后生成 v3 replay。
一次学习的 critic 更新数不超过新接纳转移数和 `max_updates_per_attempt` 两者中的较小值。
候选完整保存、重载且预测核验通过后，下一次尝试才换用其冻结快照；父模型和可靠默认版本不变。
当前最多 10 次尝试、每次 600 秒，并受已有图像和经验容量限制；这些是软件上限，不是实机资格。

外部停止、`stop.request` 或停止回调会阻止后续采样和学习；发送失败、释放未确认、无合格经验同样停止循环，
保留已启动尝试。环境级关闭成功不能覆盖某次运行的释放失败。故障记录不会生成驾驶进步结论。
此独立入口尚无专用 CLI，也不恢复被中断的异步循环；已封存候选仍可使用已有 `SACResume`，
需要阶段接续时使用版本 3 的 `LearningLoop` / `LearningContinue`。

## 新经验和持续学习

新增转移先经过已有 `SACReplayPrepare` 的独立任务、奖励、动作反馈和时序检查。记录窗口截断保留末状态 bootstrap；确认失败保留负回报与终止；接口故障不当作驾驶失败学习。

`SACResume(..., additions=((replay_path, sha256), ...))` 显式追加经验。像素、历史、任务状态口径及任务/道路/奖励摘要须一致，实际命令仍须落在已保存的动作支持范围。原始 RGB、各来源 replay 清单及排除诊断一并封存；合并后的转移以来源摘要和原转移编号标识，避免局部编号冲突。旧候选及原 replay 不改写。

```powershell
uv run --locked fh5 sac-resume --checkpoint runs/sac-001 --output runs/sac-002 --steps 8 --add-replay runs/new/prepared/replay.json <SHA256>
```

一次最多追加 10 份，追加操作最多每个新接纳转移获得 1 次 critic 更新；actor 沿用已有累计更新相位。以原始包流摘要识别已接纳的录制，同一录制重新审核或重新排版不能再次取得额度；需修改既有录制的标签时，应另做显式数据迁移，不走新增交互额度。无追加参数时保留原有固定经验续训语义，可做离线训练对照；自动循环总是根据本轮新接纳数量限定更新数。编码器更新后重算特征，不保存跨版本 latent。

## 预算、停止与结果

一次运行限 1–10 次尝试，每次 1–1000 个同步动作；图像历史总预算每次 512 MiB，累计 replay 最多 10000 个转移。经验沿用 512 MiB 像素和 128 MiB 清单上限，来源清单另外限 128 MiB/1000 份。尝试的图像预算在采样前检查；追加经验达到容量上限时，保留采样记录并停止新更新。结果目录保存所有尝试和候选，因此磁盘总量可高于单份预算。

在输出目录写 `stop.request`，采样在完整响应边界停止，学习在完整优化器更新边界保存。组合调度也可以通过实验入口的 `sac_stop_requested` 传入停止检查；`expected_checkpoint_sha256` 绑定首轮模型身份。当前外部同步调用不能被抢占；强杀只保留此前封存的候选，不声称能恢复未保存的尝试或游戏状态。采样后未进入学习的数据保留供排查。

`summary.json` / `report.html` 汇总候选、尝试、有效/排除转移及故障；每次尝试保留 `sampling.json`、`trace.json`、数值帧、完整遥测和独立结算。`candidate-NNN/` 支持已有续训入口，搬走采样目录仍可继续其封存经验。原始来源清单用于追溯，运行所需数值帧按内容摘要保存在合并经验中。

[示范/在线混合](sac-mixture.md)、[临时 BC 约束退出](sac-imitation.md)、[冻结 SAC 评估](evaluation.md)、[候选比较](candidate-selection.md)、[完整归档恢复](candidate-archive.md)及[合成版本历史/回退](candidate-store.md)已有各自的软件实现。[连续学习调度](learning-loop.md)正在接入这些接口，保留已封存阶段与独立版本角色。完整故障恢复、原生异步时序、自动重新入场及真实驾驶收益仍需验证；不能用合成回报或 loss 变化宣布车辆已进化。
