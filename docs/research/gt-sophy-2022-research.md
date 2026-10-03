# GT Sophy 2022：持续强化学习的参考边界

核查日期：2026-09-30。参考 Wurman 等人的 *Outracing champion Gran Turismo drivers with deep reinforcement learning*，Nature 602, 223–228，DOI `10.1038/s41586-021-04357-7`。本文只讨论 2022 年论文及同年作者技术说明，不混用后来基于图像的 GT Sophy 工作。

## 结论

本项目应借鉴其“实际交互 → 经验重放 → 策略更新 → 独立评估”的学习闭环。BC 是起步工具，后续轨迹预测是可选辅助任务；二者都不应把用户示范固定为驾驶上限。这里记录研究依据和建议，不替换当前 SAC 方案，不新增实机起步的前置任务。

## 论文实际做了什么

| 问题 | 核查结果 |
| --- | --- |
| 是否依靠人类动作示范 | 策略从零通过 RL 交互学习；不是先模仿冠军动作。专家仍参与场景、奖励和候选评估设计。 |
| 观测是否来自截图 | 使用游戏 API 向量状态及赛道地图，包括本车状态、局部道路点、附近对手状态。2022 年工作没有图像感知管线。 |
| 航点是否为预测输出 | 左右边界和中心线的局部点是地图产生的输入，不是监督预测的人类赛车线。 |
| 控制输出 | 直接输出转向与有符号纵向动作：正值油门、负值刹车。没有必须经过“航点预测 → 目标速度 → 控制器”的结构。 |
| 实时性 | 决策为 10 Hz；作者也测试更高频率，没有得到显著收益。这个结果不能直接当作 FH5 所有观测和车速条件的保证。 |

依据：[作者公开论文 PDF](https://sam.barrettnexus.com/publications/papers/nature22.pdf#page=2)，正文 Approach；PDF 第 7 页 Methods / Actions、Features，第 10 页 Data availability；[作者训练说明](https://ai.sony/blog/how-to-train-your-race-car)。

## 强化学习的训练信号

QR-SAC 在 SAC 基础上使用分位数价值分布和多步回报。Actor 根据观测采样动作，critic 学习动作之后的累计回报；actor 的训练目标来自 critic 与熵项，不是与某个人类方向盘标签做误差最小化。分位数预测的是**回报分布**，不是未来航点或速度。

补充材料的 Listing 1–2 展示了训练循环：采样多步交互记录，更新两套 critic、actor 和 target critic，再向采样端分发策略参数。不同场景的经验表按指定比例组成批次。启动阶段甚至使用均匀随机动作采集初始经验；这描述论文实验，不是建议我们直接对 X 999 执行随机满幅动作。依据：[补充材料 PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-021-04357-7/MediaObjects/41586_2021_4357_MOESM1_ESM.pdf#page=3)，PDF 第 3–7 页 Listing 1–2、第 8 页任务工厂。

奖励结合赛程进度、离路、撞墙、轮胎滑移，以及对手阶段的相对进度和碰撞惩罚。离路时不发正常进度奖励；不能把“油门更大、速度更快”直接当作目标。依据：[论文 Methods / Rewards](https://sam.barrettnexus.com/publications/papers/nature22.pdf#page=7)。

作者曾发现，过度精细地判断碰撞责任会让策略钻规则漏洞；最后保留轻量的任何碰撞惩罚，再对追尾等行为额外处罚。只用内置 AI 或过度激进的对手也会造成不同失败，因此混合、筛选对手群体。依据：[作者关于 sportsmanship 的说明](https://ai.sony/blog/dont-cross-that-line-how-our-ai-agent-learned-sportsmanship)。

## 采样和评估比“换一个算法名”更重要

训练覆盖空场、多车、起跑和尾流超车等场景，并随机化位置、速度和对手。混合场景持续共同训练；不是学完一个技能就永久移除。多表经验重放用于减少旧技能被遗忘，候选还要参加固定技能测试、策略对抗及人工实测。自主产生训练数据不等于研究流程没有人类设计和评审。依据：[作者训练说明](https://ai.sony/blog/how-to-train-your-race-car)，Training Regimen 与 Learning to Do It All；[论文 Methods / Training scenarios、Policy selection](https://sam.barrettnexus.com/publications/papers/nature22.pdf#page=9)。

论文接口没有在零售游戏中开放。研究能配置起始场景并利用多辆车并行采样；普通 FH5 Data Out 与 DXGI 没有因此获得对手精确状态或任意恢复游戏状态的能力。依据：[论文 Code availability](https://sam.barrettnexus.com/publications/papers/nature22.pdf#page=10)。

算力比较时必须计入采样：作者说明空场训练使用 10 台 PS4，每台 20 辆车，游戏仍按实时速度运行。新车与新赛道实验说明其训练方法可复用，不构成同一个未经再训练的策略能驾驭任意路线的证据。依据：[作者关于速度训练的说明](https://ai.sony/blog/training-the-worlds-fastest-gran-turismo-racer)，How did we train 与 Pushing GT Sophy to the limits。

## 对本项目的建议

以下是结合本项目接口与验收条件作出的设计判断，不是论文已经验证过的 FH5 结论。

1. **先让学习策略真实驾驶。** 沿现有图像管线任务完成可观测的短段闭环，确认观测、实际动作与结果对应。预测头不阻塞这一步。
2. **示范作为启动资源。** 人类实际轨迹可生成初始未来航点和速度标签；随后 AI 交互同样能产生未来状态标签，但失败、倒带和重开必须有边界及标记。
3. **把预测任务与优化目标分开。** 预测“本次后来怎样”帮助表征学习，不能直接证明“应该这样开”。不能把所有自驾片段无差别回灌成正确动作，否则可能强化自身错误。RL 用有效进度、完成情况和违规约束学习更好的决策，允许偏离人类初始路线与速度。
4. **持续学习不等于单调进步。** 保留固定的冻结评估、可靠版本与探索候选；按可比较结果晋升，不能因训练继续运行就自动替换默认策略。训练批次覆盖旧技能和新失败，监测遗忘。
5. **先测采样能力，再扩算法。** 用实际每小时有效交互量、重开成功率和训练稳定性决定扩大实验。QR-SAC 可作为后续对照；当前不因论文成绩直接替换 SAC，也不照搬其 GPU/PS4 集群预算。

当前产品和算法基线仍以[驾驶学习主方案](../design/driving-learning-design.md)、[奖励与有效性设计](../design/reward-and-validity-design.md)为准。
