# CPU BC 中断与续训

`collection-bc-train` 的 CPU 数值 BC 在协作式停止边界保存 learner 检查点。
它包含驾驶网络、完整 Adam 状态、Torch 随机数状态和累计更新数，区别于只供推理的 `actor.pt`。
资源压力停止后，可在新目录执行原实验预算中尚未完成的更新：

```powershell
$checkpointHash = (Get-FileHash -LiteralPath runs/scheduled-candidate-001/learner/learner.json -Algorithm SHA256).Hash.ToLowerInvariant()
uv run --locked fh5 collection-bc-resume --run runs/scheduled-candidate-001 --output runs/scheduled-candidate-002 --checkpoint-sha256 $checkpointHash
```

恢复优先读取父运行的 `learner/learner.json`，因此 `schedule.json` 缺失或截断时也可用上述命令。
该命令对本地已封存清单计算摘要；若已有留存摘要，应使用留存值核验原件。
未成功发布新 learner 的失败运行可以通过完整的 `schedule.json` 指向仍有效的祖先检查点；
此时使用报告中的 `learner_checkpoint.manifest_sha256`，也可以直接从原父运行恢复。

输出目录必须是新目录，不能位于父运行、父检查点、数据集或采集环境内。
新运行读取父目录的冻结调度配置；训练配置、数据摘要、总更新预算、采样顺序和模型输入契约保持一致。
不增加新的 `--steps` 预算，也不靠重新设种子或重跑旧更新来模拟续训。
构造并加载网络及优化器后，才恢复随机数状态。

## 保存与结果

- `learner/learner.pt` 保存必要训练状态；`learner/learner.json` 在权重写入、刷新及校验后最后发布。
  不覆盖父检查点。增长的组清单和来源历史留在原数据集中，不复制到每一代 learner 元数据。
- `steps_completed` 是实际完成的累计更新数，`durable_steps_completed` 是已封存的进度；
  `steps_this_run` 只统计本次新增更新。保存失败时，不能将尚在内存中的更新报成已持久化。
- `learner_checkpoint` 与 `candidate` 分开。训练状态可先保存；只有完成冻结预测核验和发布后才产生候选。
  原更新预算已完成时，恢复只补做验证和发布，新增更新数为零。
- loss 日志是可选诊断，缺失不会阻止续训。每次恢复产生本次运行的日志，步号沿用原累计进度；
  不把局部日志伪装成全部训练历史。
- 累计耗时和梯度统计缺失或无效时，必要训练状态仍可接续；learner 的 `statistics.status`
  及候选的 `training.statistics_status` 标为 `partial`，不将本次可用统计冒充完整历史。
  文件摘要、必要元数据、网络、Adam 和随机数状态仍须通过完整性核验。
- 调度报告写入遇到 `OSError` 或 `MemoryError` 时，已完成的学习与候选发布结果保持完成；
  返回摘要 `learning_schedule.schedule_report.status` 为 `unavailable`，`report_path`
  指向已封存的 `learner/learner.json`。错误说明保留在返回摘要中，不能假设失败的报告已写入磁盘。
- 抽样之后、Adam 更新之前发生像素读取或内存分配错误时，释放未完成批次，恢复抽样前的随机数状态，
  保存上一个完整更新边界。恢复后重新读取并校验输入，不跳过失败批次。
  更新边界的资源探测遇到同类错误时，也先保存完整边界，再报告原错误。

## 当前边界

当前支持 CPU temporal BC 的完整更新边界，检查 Torch 版本、CPU 线程数和确定性运行契约。
恢复还核验 Adam 参数归属、全部历史、冻结更新选项、矩张量形状/类型和累计更新计数；
即使重新绑定文件摘要，不兼容的训练状态也在启动学习前拒绝。
原始数据集及其数值像素文件必须继续可用且通过完整性检查；父运行和冻结配置也须保留。
这不是可任意移动、完全自包含的训练包。

协作式停止由调度检查点响应；强制终止进程、断电、Adam 执行中断不具备逐步事务恢复保证。
Ctrl+C 不等于已保存当前内存状态，恢复以实际发布的 learner 为准。
旧 actor-only 模型、legacy BC 和 CUDA 中途续训不在此入口的已验证范围内。

公开 CPU 测试已证明：连续 6 次更新与停止后 2＋4 次更新的模型、Adam、随机数及预测精确一致；
删除可选 loss 日志仍可接续；已完成 3 次更新的父检查点通过命令行恢复时新增零次更新。
候选目录发布失败、learner 权重写入失败和最终清单发布失败均有公开文件故障验证；
必要保存失败时保留旧父状态，摘要区分已计算与已封存进度，之后可再次接续。
像素读取及资源探测的 OS/内存故障、可选统计缺失/无效、报告缺失/截断也已有专项证据。
独立审阅和完整回归尚未完成，结果与边界见[资源清理记录](validation/resource-limit-audit.md)。
这些是软件训练状态证据，不代表实机驾驶或 4K 游戏与训练并行验收。
