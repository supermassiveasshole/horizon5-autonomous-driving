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

## 新经验和持续学习

新增转移先经过已有 `SACReplayPrepare` 的独立任务、奖励、动作反馈和时序检查。记录窗口截断保留末状态 bootstrap；确认失败保留负回报与终止；接口故障不当作驾驶失败学习。

`SACResume(..., additions=((replay_path, sha256), ...))` 显式追加经验。像素、历史、任务状态口径及任务/道路/奖励摘要须一致，实际命令仍须落在已保存的动作支持范围。原始 RGB、各来源 replay 清单及排除诊断一并封存；合并后的转移以来源摘要和原转移编号标识，避免局部编号冲突。旧候选及原 replay 不改写。

```powershell
uv run --locked fh5 sac-resume --checkpoint runs/sac-001 --output runs/sac-002 --steps 8 --add-replay runs/new/prepared/replay.json <SHA256>
```

一次最多追加 10 份，追加操作最多每个新接纳转移获得 1 次 critic 更新；actor 沿用已有累计更新相位。以原始包流摘要识别已接纳的录制，同一录制重新审核或重新排版不能再次取得额度；需修改既有录制的标签时，应另做显式数据迁移，不走新增交互额度。无追加参数时保留原有固定经验续训语义，可做离线训练对照；自动循环总是根据本轮新接纳数量限定更新数。编码器更新后重算特征，不保存跨版本 latent。

## 预算、停止与结果

一次运行限 1–10 次尝试，每次 1–1000 个同步动作；图像历史总预算每次 512 MiB，累计 replay 最多 10000 个转移。经验沿用 512 MiB 像素和 128 MiB 清单上限，来源清单另外限 128 MiB/1000 份。尝试的图像预算在采样前检查；追加经验达到容量上限时，保留采样记录并停止新更新。结果目录保存所有尝试和候选，因此磁盘总量可高于单份预算。

在输出目录写 `stop.request`，采样在完整响应边界停止，学习在完整优化器更新边界保存。当前外部同步调用不能被抢占；强杀只保留此前封存的候选，不声称能恢复未保存的尝试或游戏状态。采样后未进入学习的数据保留供排查。

`summary.json` / `report.html` 汇总候选、尝试、有效/排除转移及故障；每次尝试保留 `sampling.json`、`trace.json`、数值帧、完整遥测和独立结算。`candidate-NNN/` 支持已有续训入口，搬走采样目录仍可继续其封存经验。原始来源清单用于追溯，运行所需数值帧按内容摘要保存在合并经验中。

仍需继续：示范/在线数据混合与临时模仿约束退出、冻结 SAC 评估接入、候选筛选、原生异步时序和自动重新入场、完整在线学习调度恢复，以及真实驾驶收益。不能用此处的合成回报或 loss 变化宣布车辆已进化。
