# T14 / #15：环境获取的有限重试

日期：2026-10-01。此增量处理采样/评估环境尚未获取成功时的暂时不可用；不覆盖已开始驾驶的重跑、菜单失败或未封存任务恢复。范围与配置见[连续学习循环](../learning-loop.md)。本票保持开放。

## 原型证据

全部通过公开 `run_experiment` / `LearningContinue` 入口，使用合成外部环境。下表原型复用已保存的真实 CPU learner，不重新训练，也没有替换内部模型、优化器或验证结果。每次使用新的输出目录和可写版本库/登记库副本，原始模型及学习记录保持只读。

| 场景 | 结果 | 本机证据 |
|---|---|---|
| 显式重试预算 | 原先不接受该配置；新增后 2 次重试对应最多 3 次请求，耗尽后停止，0 次更新、无子任务 | `runs/t14-acquisition-budget-{red,green}-20261001/` |
| 获取返回时收到停止 | 原先留下空采样目录；修复后关闭连接、不创建子任务 | `runs/t14-acquisition-stop-{red,green}-20261001/` |
| 已训练候选等待评估 | 原先第一次获取失败即结束；新增后最多 3 次请求，同一 learner 仍为 3 次更新，原学习摘要不变 | `runs/t14-evaluation-acquisition-{red,green}-20261001/` |
| 清理抛出同类型异常且停止文件被外部解除 | 原先错误地再次获取连接；修复仅捕获获取调用的异常，关闭故障停止，只创建 1 个连接 | `runs/t14-acquisition-close-{red,green}-20261001/` |
| 未释放、普通异常、未配置重试 | 各自仅调用一次；整体关闭成功不能遮蔽子资源释放失败 | `runs/t14-acquisition-{release_failed,ordinary_error,no_retry_configuration}-20261001/` |
| 等待期间用户停止 | 5 秒等待被中断，收到停止后 2 秒内返回，无第二次请求 | `runs/t14-acquisition-wait-latency-20261001/` |
| 重试前磁盘余量下降 | 再次容量核对拒绝，停止于 `storage_budget_exhausted`，仅 1 次请求 | `runs/t14-acquisition-capacity-20261001/` |
| 耗尽后显式接续 | 保留旧失败历史与 learner，后续可以重新获取；测试随后主动停止，未开始训练 | `runs/t14-acquisition-continue-20261001/` |

评估原型使用 `runs/t15-capacity-fresh-20261001/test_capacity_stop_after_learn0/loop/` 中真实的评估前 learner/学习记录，在隔离夹具中增加新的重试配置及绑定摘要。它验证新输入下的行为，不宣称允许在生产接续时修改原冻结配置。正式整文件测试从头生成同类状态。

配置反例通过：`tests/test_learning_retry.py::test_unbounded_or_ambiguous_retry_policy_is_rejected_before_opening` **3 passed / 1.24 秒**。过大次数、无穷等待和布尔次数在创建输出目录/连接前拒绝。Ruff、格式（286 文件）、严格类型检查（115 源文件）通过。

## 待完成验证

`tests/test_learning_retry.py` 共 14 项，包含从头生成实际 CPU 模型、采样与评估首次获取分别失败后自动完成一轮的验证。整文件尚未运行；先等待主分支 `feb1ca2` 的完整回归释放训练/写盘测试资源。主分支那次回归不覆盖本增量。

Standards / Spec 审查和增量完整回归待完成。以上原型不证明真实游戏可恢复或驾驶能力改善；没有启动 FH5、Steam、原生捕获或控制器。

## 剩余范围

重试次数按一次运行或显式接续计量，循环不自动重启一个耗尽预算的会话。等待可响应停止；外部获取调用本身仍须有界，本机制不抢占阻塞的适配器。已开始的驾驶、菜单失败、未封存阶段恢复以及真实游戏长期循环另行验证，不覆盖旧失败来获得成功结果。
