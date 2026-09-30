# SAC 数值转移、预热与离线策略更新

对应 [#11](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/11) 的离线软件实现。当前提供数值转移构造、完整冻结 BC 的双 Q 预热，以及实际更新 actor、双 Q、温度和共享编码器的有界 SAC 训练。输入仍为明确声明的合成经验，尚未形成自主游戏采样循环或证明驾驶改善。所有入口均不连接游戏或控制设备。

## 输入与边界

`run_experiment(SACReplayPrepare(...))` 将录制、独立局部任务及有效性证据、奖励配置和合成动作记录关联起来。记录格式为 `synthetic-synchronous-action-trace-v1`：明确声明动作在起始遥测时刻同步生效。它只接受 `source_kind=synthetic` 的录制，不能把原生下发返回时间解释成游戏实际执行时间。

记录含起始命令、下发时刻、动作所属 epoch、每段起止包号、发送状态、实际整数命令和数值观测。油门/刹车互斥；转向使用 `steer_i16 / 32767`，纵向使用 `(throttle_u8 - brake_u8) / 255`。观测含 RGB 原始字节及摘要、采集/就绪时间、因果本车状态和动作历史。当前适配器固定动作历史偏移为 200/100/0 ms；同一决策的新命令不能进入自己的历史。测试中的 `tests/test_sac.py::experience` 是完整合成格式示例。

构造器核验数值帧、遥测、实际发送值与合成响应的一致性，复用独立奖励结算。缺图、未发送、归属不符和跨恢复边界会保留排除原因，不填帧或拼接重开后的状态。真实任务失败保留负奖励和终止；外部录制结束仅在真实末观测存在时继续 bootstrap，不能自动当成零价值终止。

恢复后的完整历史同样受独立前向片段约束：epoch 必须更新，帧来源和动作历史不得早于恢复边界。刚恢复时历史不足的决策跳过；凑齐恢复后的真实历史即可继续构造转移，不必丢弃整段。

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
uv run --locked fh5 sac-train --config configs/sac-learning.example.json --output runs/sac-candidate
uv run --locked fh5 sac-policy-replay --checkpoint runs/sac-candidate --replay runs/sac-replay/replay.json --report runs/sac-policy.html
```

`sac-prepare` 无可用转移时返回 4；输入错误返回 2；成功返回 0。报告展示资格、排除原因、Q 更新量与 BC 不变检查。loss 或 Q 变化不等于驾驶进步。

## 策略、温度与共享编码器更新

`SACTrain` 从已保存的 BC/双 Q 预热检查点出发，采用固定经验和有限 CPU 更新预算。`steps=0` 可单独核对交接，不执行优化。新策略版本 `conditional-temporal-sac-v1` 保留原数值 Δt 输入，并向策略提供上一实际命令、实际间隔和可执行区间；任务进度/计时上下文仍只提供给 critic。

策略把 BC 的受限动作转换为有限逆 tanh 均值，新增动作上下文均值修正和 log 标准差。初始修正为零，默认 log 标准差 −3，范围 [−5, −1]，以较窄高斯开始探索；饱和边界的均值留在同一整数命令格内。保存前报告所用经验上的 BC 交接整数命令误差。此检查不证明所有未知状态或实机驾驶相同。

采样 `z = mean + std * noise`，连续动作 `a = center + scale * tanh(z)`。log 密度包含正态项、tanh Jacobian 和 `log(scale)`；命令坐标下目标熵为归一化目标熵（默认 −2）加两轴 `log(scale)` 之和。温度优化使用相同坐标。真实终止的目标只含结算奖励；非终止采用 `r + discount * (min(target Q) - alpha * log_probability)`。

最终发送契约仍是整数命令。Q 前向及 replay 使用量化后的值；actor 对量化使用**直通梯度近似**，熵密度定义在量化前的连续动作上。这是版本化的连续松弛，不把整数命令的概率质量冒充连续密度，也不宣称是精确离散 SAC。后续若改变近似或坐标，须更换契约并重新对照。

critic 优化器独占图像/状态编码器和双 Q；actor 优化器只持有策略头及上下文/方差层，actor 在共享特征处停止梯度；目标网络有单独的目标编码器。每次 critic 更新后重新编码 actor 输入，训练不缓存旧 latent。每两次 critic 更新一次 actor/温度，编码器默认学习率 1e−5，actor 3e−5，critic/温度 1e−4，目标软更新率 0.005。报告检查双方不越权修改参数。固定经验上的更新次数不等于在线采样更新比；有界新采样调度仍待接入。

`policy.json` 与 `policy.pt` 相互绑定配置、动作语义、BC、经验和权重摘要，保存两套优化器、温度优化器、目标网络、RNG 和步数；目前用于冻结重载检查，完整跨进程续训由 #13 接入。独立输出保留初始化 BC，原检查点不会覆盖。原始 uint8 图像按内容去重，上限 512 MiB；单次浮点图像 batch 上限 256 MiB。当前仅验证 CPU；GPU/游戏共同运行预算未验证。

`SACPolicyReplay` 只接受新的 `.html` 输出路径，既有文件和符号链接在计算前拒绝，最终独占创建报告；不能用回放报告覆盖候选权重、经验、数值图像或已有报告。实际更新与保存重载的证据见[本次验证](validation/t10-sac-updates.md)。

## 后续工作

接入有界采样/学习交替、尝试边界换版、示范混合与模仿约束退出。原生动作时序适配、完整数据用途登记、GPU 与 4K 游戏共存、实机驾驶比较仍待完成。#11 保持开放，这些实机项不阻塞可独立测试的学习循环、#13 续训和 #14 筛选实现。
