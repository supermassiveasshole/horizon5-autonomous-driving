# 临时 BC 约束与开发评估驱动退出

对应 [#11](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/11)。
在普通双 Q SAC 上可显式加入原始冻结 BC 的动作约束，并根据训练前固定的开发评估协议逐级减弱至零。当前开发门槛只支持明确声明的合成环境；不会激活真实默认驾驶版本，也不证明 FH5 驾驶改善。原生时序、真实数据谱系和实机独立评估仍待接入。

## 优化目标

`imitation_weights` 是严格递减、最后为零的系数列表，长度 1–16，每项有限且处于 `[0, 10]`，例如 `[1, 0.5, 0]`。省略该字段继续原有纯 SAC；`[0]` 显式表示已退出，模型状态和随机更新与纯 SAC 相同。

每次 actor 更新，在当前抽到的观测上分别计算原始 BC 与当前策略的确定性命令。两者使用同一可执行区间和整数命令网格，动作差除以每轴区间宽度后求均方，乘当前系数加到 SAC actor loss。量化使用与 Q 动作相同的直通梯度近似。

这约束的是当前状态上的动作偏离，不是人类历史赛车线或速度曲线。完整示范转移的 replay 配额与此系数不同，仍见[示范混合](sac-mixture.md)。critic 继续独占共享编码器，actor 在共享特征处停止梯度；整个教师 BC 保持冻结。零系数不再调用教师计算约束。

## 冻结与评估

在首次训练配置中提供 `imitation_protocol_batch`，指向通过 `evaluation-prepare` 冻结的 **development** 批次。保存车辆/辅助、运行时、任务资产、各参考条件尝试数及可靠性/用时门槛的摘要。路径相对于训练配置文件。未提供协议时允许固定系数训练，但不能事后添加退出协议。

```json
{
  "version": 1,
  "warmup": "../runs/critic-ready",
  "replay": "../runs/critic-ready/experience/replay.json",
  "steps": 100,
  "imitation_weights": [1, 0.5, 0],
  "imitation_protocol_batch": "../runs/development-protocol"
}
```

每轮分别为原始 BC 与当前精确 SAC 检查点预先登记新开发批次，关闭探索和倒带；执行后保留完整 ledger、原始执行与数值图像、起跑及独立有效性依据。比较文件采用 [CandidateCompare](candidate-selection.md) 的格式，绑定双方批次和 ledger 摘要。

```powershell
uv run --locked fh5 sac-train --config configs/my-guidance.json --output runs/guided
uv run --locked fh5 sac-resume --checkpoint runs/guided --output runs/reviewed --steps 0 --imitation-comparison runs/comparison.json --imitation-registry runs/usage.sqlite
uv run --locked fh5 sac-resume --checkpoint runs/reviewed --output runs/continued --steps 100
```

审核会重算源记录，而不是读取可手填的“passed”或既有 selection 结论。须满足：

- 当前检查点和原始 BC 精确匹配，协议与训练前一致；正式最终验收数据不能反馈给课程调整。
- 全部计划尝试保留，起跑证据完整，数值重放核实真正的 BC/SAC 执行，拒绝仅诊断预测的替代品。
- 按原协议先判断合法性，再比较可靠性与各参考条件用时；候选至少一组达到规定改善，其他组不退步。
- 开发记录在用途库中预先登记、来源完整且无已知重复；不得与已封存 SAC 经验或该候选此前使用的评估记录重叠。

当前门槛明确限定为 **synthetic_development_only**。用途库与原始摘要只检查已登记历史，不能证明未登记或变换后的全部历史独立；合成 BC 谱系声明也不是实机资格。报告保留这些限制，不改变既有 `automatic_promotion_allowed=false`。

一次达标最多前进一个系数阶段；下一次必须评价新的精确检查点、使用新收集的两组记录。未达标保留系数、候选和完整失败报告，仍可继续 SAC 更新。训练 reward、loss 和更新次数不触发阶段变化。

## 保存和继续

采用约束的检查点版本为 4，保存教师、协议、阶段、系数、评估来源和审核报告摘要。审核报告随检查点复制，也随冻结评估模型复制。普通续训继承全部状态；退出到零后不会自动恢复约束。

`steps=0` 只审核并封存，不改变学习器参数、优化器或 RNG。最多保留 64 次阶段审核，审核文件合计上限 128 MiB；超限应停止并保留既有检查点。重新核验原始评估仍需要保存比较文件所引用的源记录，检查点内的审核摘要不能替代它们。

软件验证从已约定的 `run_experiment` / CLI 入口进行，包含真实 CPU 网络更新、模型命令驱动的合成短程评估及续训，见[验证记录](validation/t10-sac-imitation.md)。不会启动游戏或原生控制设备。整票 #11 仍保持开放，后续继续真实动作时序、新数据和驾驶效果验收。
