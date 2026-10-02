# 有限 Q 预热的停止与恢复（T12 / #13）

`SACCriticWarmup` 冻结整个 BC，只训练双 Q；`SACCriticResume` 继续尚未完成的同一阶段。两者均经 `run_experiment` 或 CLI 调用，在 CPU 上读取封存的合成数值经验，不连接游戏或控制设备。SAC 阶段的恢复见 [SAC 中断与续训](sac-resume.md)。

## 固定预算与操作

```powershell
uv run --locked fh5 sac-warmup --model runs/temporal-bc --replay runs/sac-replay/replay.json --replay-sha256 <摘要> --output runs/critic-001 --steps 100
# 训练开始后，在输出目录创建 stop.request 请求协作停止。
uv run --locked fh5 sac-warmup-resume --checkpoint runs/critic-001 --output runs/critic-002
```

首次 `--steps` 为整个预热阶段的正整数实验预算，不叠加任意次数上限。恢复默认只完成剩余步数；可选 `--steps N` 限制本段执行数量，不能超过剩余预算。恢复时 `--steps 0` 只核验并导出快照。阶段已完成时再次恢复执行零次更新，不重跑 BC、不重新初始化 Q，也不隐式延长预热。

输出目录必须全新且位于源模型、经验及检查点之外。`--checkpoint-sha256` 可绑定预期父清单摘要。batch、学习率、种子、动作包络及总预算沿用原配置，当前目标软更新率固定为 0.005。

停止文件与实验入口的 `sac_stop_requested(completed)` 在更新边界检查；回调参数是该预热阶段累计完成步数。停止时保存完整状态、返回退出码 4；本段预算完成返回 0，输入错误返回 2。返回 0 不一定代表整个阶段完成，应检查报告的 `phase_status`、`total_steps` 和 `warmup_remaining_steps`。只有阶段完成的版本 2 预热快照能交接给 `SACTrain`。

## 可恢复快照

新 `critic.json` 为版本 2，绑定以下依赖：

- `critic.pt`：双 Q、目标 Q、完整冻结 BC 状态、Q 优化器、CPU RNG 和累计步数；学习状态另有规范化摘要。
- `actor/`：未改变的 BC 权重与输入、归一化和模型清单；恢复时重新编码数值帧，不混用另一版编码器缓存。
- `experience/`：原始 RGB 字节、经验及来源清单，保留动作、时序、Δt、奖励和任务边界；无需原工作区路径。
- `training-report.json`、`history/`：本段与全部祖先的清单及训练报告。交接给 SAC 后保留预热历史，后续 SAC 续训继续归档。
- `resume_contract`：阶段、优化器归属、固定预算、CPU/Torch 版本、两线程及确定性设置。模仿衰减、辅助预测头尚未采用，不能把缺省视为已支持。

载入时核对权重、经验、BC、报告、历史及契约。损坏或不兼容的输入拒绝发布新快照；`critic.json` 最后写入，失败目录不算可恢复结果。预热与 SAC 共用流式历史校验/复制，权重验证私有副本后加载；检查点权重、清单、报告及祖先历史不再因固定文件大小或条数被拒绝。

祖先清单改为与 SAC 共用的[摘要关联历史节点](sac-resume.md)，权重与清单仅绑定链头和条数。旧数组格式可以读取，续训输出转换为新格式，原快照保持不变。预热交接 SAC、归档和恢复均保留这些节点及其原始证据。

每步更新明细改为[可选 JSONL 日志](training-diagnostics.md)，报告仅保存其描述，不累计 loss/目标数组；诊断失败不丢弃训练，丢失日志仍可续训。原始报告字节及旧数组格式保持可读。

依据[资源约束](resource-policy.md)，正常增长不能用任意数字阻止续训。经验清单、封存帧、batch 等旧限制及按转移展开的预测/特征仍待清理，见[审计](validation/resource-limit-audit.md)。这些遗留项不构成容量依据，不能要求用户单纯缩短训练以迁就它们。实际写入失败保留父快照，可选报告失败不撤销完成的训练。

旧版本 1 保留冻结预测和原有 SAC 初始化能力，但缺少封存经验与完整历史，明确拒绝预热续训。`sac-critic-replay` 只接受全新的 `.html` 输出，不能覆盖权重或已有报告。

## 验证边界

[跨进程原型与故障验证](validation/t12-critic-resume.md)使用真实 CPU Torch 更新和合成经验。协作停止不抢占一次优化器操作；强杀只能回到最后已封存快照，首次运行尚无快照时不能保住中间更新。当前不恢复游戏位置、输入租期或图像缓存，也不证明 FH5 驾驶改善。#13 的实机重新入场、后续实际采用的模仿调度及可选组件仍待完成。
