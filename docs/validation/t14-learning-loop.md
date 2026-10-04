# 学习循环与恢复的软件证据

本页收拢 2026-10-01 至 2026-10-02 的合成循环、有限重试和父状态恢复验证。通过公开 `run_experiment` / `LearningContinue` 入口，使用实际 CPU 模型、优化器、冻结评估和候选库；外部 I/O 为合成环境，文件发布故障由真实子进程退出或外部文件系统注入。未启动 FH5、原生捕获或控制器。

当前操作与恢复规则见[连续学习循环](../guides/sac/learning-loop.md)。后续异步及原生适配器的软件证据见[异步连续学习记录](t14-asynchronous-learning.md)，最新整仓结果和资源边界见[资源审计](resource-limit-audit.md)。以下结果仅对应各自版本，不重复累加，也不代表实际驾驶改善。表中目录为历史执行路径；本次保留结果摘要及验证 JSON、JUnit、日志，可重建的 pytest 临时夹具可以清理。

## 循环、提交与评估恢复

| 已验证行为 | 保留证据 |
|---|---|
| 两轮实际采样与 SAC 更新，累计从 30 增至 36 步；候选缺独立合法性证据时保留默认版，继续探索版 | 初始提交 `a9edd25`；相关 67 项通过。原型 `runs/t14-resume-green-20261001/`、`runs/t14-eval-stop-green-20261001/` |
| 子资源释放失败不得被整体关闭掩盖；原始采样、模型或审阅附件变化时拒绝继续 | `runs/t14-menu-release-green-20261001/`、`runs/t14-rejected-release-green-20261001/`、`runs/t14-originals-green-20261001/` |
| 候选提交前/后退出，接续只完成未记账工作，不重复采样、更新或发布；不匹配请求、模型或版本库提交拒绝 | `runs/t14-commit-crash-green2-20261001/`、`runs/t14-precommit-crash-green-20261001/`、`runs/t14-commit-crash-rejection-20261001/`；修复 `397f760` 的准确中断阶段验证：`runs/t14-recovery-diagnostics-20261001-a/`，4 项通过 |
| 成功子采样封存、父状态未登记时接纳原 learner；漏掉像素或审阅附件仍拒绝 | `runs/t14-sealed-sampling-green-20261001/`、`runs/t14-sealed-sampling-rejection-20261001/`、`runs/t14-source-inventory-{red,green}-20261001/` |
| 评估完成后父状态退出，核对原模型、配置、输入、日志及菜单后登记；不补跑故障槽位，不把未开始项当成功 | `runs/t14-sealed-evaluation-{red,green2}-20261001/`、`runs/t14-evaluation-rejection-20261001/`、`runs/t14-failed-evaluation-{red,green}-20261001/` |
| 实际菜单操作、原运行时长和唯一原始停止事件均核验；重签摘要不能代替原执行证据 | `runs/t14-start-operation-{red,green}-20261001/`、`runs/t14-evaluation-duration-{red,green}-20261001/`、`runs/t14-original-stop-green-20261001/` |

## 有限重试与父审阅恢复

| 已验证行为 | 保留证据 |
|---|---|
| 获取失败只使用显式额度；等待可响应停止，关闭失败、未释放或冻结输入变化不触发再次获取 | `runs/t14-acquisition-fresh-20261001/`，16 项通过，包含真实 3 次更新和 2 次冻结评估；修复复核至 `3039874` |
| 只对已封存、已释放、零更新且无候选的失败使用剩余重采样额度；用户停止不自行重试，显式接续不刷新额度 | `0f6fb63`：`runs/t14-resampling-fresh-20261001/`，10 项通过；停止反例 `runs/t14-resampling-child-stop-{red,green}-20261001/` |
| 父审阅发布时退出，恢复可重建派生结果但不改原评估或独立证据；原有用户停止原因保留 | `b388d0c`：`runs/t14-parent-review-regressions-20261001/`，4 项通过；原型 `runs/t14-parent-stop-{red,green}-20261001/` |
| 失败子采样已封存、父状态未登记时恢复原零更新失败；清单漏原件、摘要计数伪改或尝试目录缺失均拒绝 | `runs/t14-failed-ack-reviewed-20261001/`，当时 23 项组合通过；后补目录缺失反例 `runs/t14-failed-child-missing-attempt-{red,green}-20261001/`，复核至 `36c1dbd` |

旧记录中的重试/报告数量和文件大小门槛不能作为现行容量依据；当前预算与结构以操作指南和资源审计为准。

## 完整更新边界的恢复

| 已验证行为 | 保留证据 |
|---|---|
| 已登记 learner 中断后只补足剩余额度；原模型、优化器和预测与连续更新一致 | `ff12e0e`：`runs/t14-partial-updates-fresh-20261002-results.xml`，5 项通过，实际补齐 3 次更新 |
| 旧会话的原采样祖先须核验；尚有剩余额度却已冻结旧模型评估时，在训练及父状态改写前拒绝 | `b92fc83`：`runs/t14-update-final-{stop,audit,parent,evaluation}-20261002/`，四项零更新原型；不把它们当作新梯度验证 |
| 完整续训子结果封存、父状态未登记时只登记一次，不再采样或训练；不完整清单、错误祖先拒绝 | `46a5b48`：`runs/t14-pending-updates-fresh-20261002-results.xml`，5 项通过，包含 3 次真实更新后子进程以 73 退出及接续 |
| 初次采样接纳经验后停止并封存完整 learner，即使启用失败重采样也保留原剩余更新，不消耗重试额度 | `8c39729`：`runs/t14-stopped-sampling-fresh-20261002-results.xml`，7 项通过，解除停止后补齐 3 次更新和 2 次冻结评估，不重新采样 |

这些恢复不推断未封存模型已经完成；必要原件损坏、资源释放不明及更新计数不一致仍拒绝。旧冻结评估与未完成更新并存的自动迁移、半次优化器更新和实际无人连续驾驶没有由这些证据验收。

## 实时边界与失败原件

紧邻 SAC 决策无可执行范围时等待，之后恢复；伪造等待上下文会被独立重放拒绝。证据为 `runs/t14-support-related-20261001/`（62 项）与 `runs/t14-support-forgery-20261001/`（2 项）。普通暂缺历史当前保持中立等待，不能沿用旧决策看门狗叙述推断必须锁止。

预览 PNG 写入曾阻塞数值原件；独立预览队列及发布/关闭同步后，`runs/t14-preview-reviewed-20261001/` 的 85 项回归通过。停止时已返回推理的封存修复见[实时基础证据](t34-realtime-foundation.md#停止时保留已返回推理)。

历史整仓失败执行目录包括 `runs/t14-sealed-sampling-focused-20261001-a/`、`runs/t14-complete-recovery-full-20261001-a/`、`runs/t14-support-final-full-20261001/` 和 `runs/t14-parent-capacity-final-full-20261001/`。当时分别记录存档缺口、SAC 动作范围或停止封存问题，失败结论未改写为成功；预览阻塞是独立复现的缺陷，未追认为所有历史写盘停顿的唯一原因。

这些软件结果不证明 4K 游戏共享负载、真实自动重开或自主驾驶收益。实际验收仍须按当前指南单独记录。
