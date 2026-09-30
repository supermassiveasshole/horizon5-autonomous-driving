# SAC 转移与双 Q 预热

对应 [#11](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/11) 的第一段软件实现。当前提供数值转移构造、完整冻结 BC 的双 Q 预热及保存重载；不是完整 SAC 策略更新，也不代表已改善驾驶。三个入口都不连接游戏或控制设备。

## 输入与边界

`run_experiment(SACReplayPrepare(...))` 将录制、独立局部任务及有效性证据、奖励配置和合成动作记录关联起来。记录格式为 `synthetic-synchronous-action-trace-v1`：明确声明动作在起始遥测时刻同步生效。它只接受 `source_kind=synthetic` 的录制，不能把原生下发返回时间解释成游戏实际执行时间。

记录含起始命令、下发时刻、动作所属 epoch、每段起止包号、发送状态、实际整数命令和数值观测。油门/刹车互斥；转向使用 `steer_i16 / 32767`，纵向使用 `(throttle_u8 - brake_u8) / 255`。观测含 RGB 原始字节及摘要、采集/就绪时间、因果本车状态和动作历史。当前适配器固定动作历史偏移为 200/100/0 ms；同一决策的新命令不能进入自己的历史。测试中的 `tests/test_sac.py::experience` 是完整合成格式示例。

构造器核验数值帧、遥测、实际发送值与合成响应的一致性，复用独立奖励结算。缺图、未发送、归属不符和跨恢复边界会保留排除原因，不填帧或拼接重开后的状态。真实任务失败保留负奖励和终止；外部录制结束仅在真实末观测存在时继续 bootstrap，不能自动当成零价值终止。

奖励按物理时间累计：多步转移使用 `r1 + d1*r2 + ...` 和折扣乘积。游戏物理时长、主机动作保持时长、图像实际帧间 Δt 分别保存。任务最远进度、起点、下一检查点和剩余时间送入 critic；BC actor 的输入契约不变。

## 预热与重载

`SACCriticWarmup` 加载数值 Δt BC，冻结整个图像编码器、状态编码器、动作层及状态缓冲，只更新两套 Q 头。目标采用受幅度、上一发送命令和实际时间限制的确定性 BC 动作，并量化至实际命令网格。超出支持区间的 replay 动作拒绝训练，不能裁剪标签后继续使用。

默认 100 次 CPU 更新、batch 32、学习率 0.0001、种子 7、目标软更新率 0.005。命令幅度默认转向 0.5、油门 0.35、刹车 0.4，两轴变化率各 4/s；可用 `--bounds` 提供对应 `ActionBounds` 字段的 JSON。配置是软件候选，不是已校准的实机策略。

动作边界先按连续坐标计算，再沿用发送器的四舍六入五成双量化（Python `round`）；例如转向上限 0.5 实际发送 16384，对应约 0.5000153。replay 支持范围使用同一实际整数端点，不能因半格量化差异拒绝真实 BC 命令。连续或量化区间退化时显式拒绝，不构造不存在的分布。

原始数值帧始终保存；预热缓存特征只属于当前冻结编码器。每次输出绑定 replay、BC 与 critic 权重摘要；保存 optimizer 和 RNG 供后续恢复接口使用。目前 `SACCriticReplay` 只加载核验，不继续训练。保存后重新加载 BC，比较同输入预测及所有原始模型状态，保证预热没有悄然改变驾驶输出。

`critic.json` 只保存小型检查点清单；逐步 loss、目标和预测位于 `training-report.json`，避免诊断数据随训练增长后超过清单读取限制。暂停/恢复入口的独立终止调整归入最后一个可用转移；新的 epoch 或独立前向片段不继承旧命令约束。

## 命令

先安装 `learning` 可选依赖。路径替换为实际合成输入；输出目录必须是新目录。

```powershell
uv run --locked fh5 sac-prepare --recording runs/synthetic/recording --trace runs/synthetic/trace.json --task runs/synthetic/task.json --reward runs/synthetic/reward.json --evidence runs/synthetic/evidence.json --output runs/sac-replay
uv run --locked fh5 sac-warmup --model runs/temporal-bc --replay runs/sac-replay/replay.json --replay-sha256 <prepare返回的摘要> --output runs/critic-first --steps 100
uv run --locked fh5 sac-critic-replay --checkpoint runs/critic-first --replay runs/sac-replay/replay.json --report runs/critic-reloaded.html
```

`sac-prepare` 无可用转移时返回 4；输入错误返回 2；成功返回 0。报告展示资格、排除原因、Q 更新量与 BC 不变检查。loss 或 Q 变化不等于驾驶进步。

## 后续工作

继续实现 tanh 高斯采样、命令坐标 Jacobian 与熵温度更新、actor 更新、critic 所属共享编码器解冻和目标编码器同步；随后接入有界采样/学习交替、尝试边界换版、示范混合与模仿约束退出。原生动作时序适配、完整数据用途登记、GPU 与 4K 游戏共存、实机驾驶比较仍待完成。#11 保持开放，这些实机项不阻塞可独立测试的学习器、#13 续训和 #14 筛选实现。
