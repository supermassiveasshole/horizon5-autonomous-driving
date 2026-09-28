# FH5 歌利亚自主驾驶与学习实验

使用用户固定调校的兰博基尼第六元素，从遥测与已知路线起步，建立能自行采样、训练、评估并改进驾驶策略的实验系统。首版聚焦无对抗歌利亚，后续探索对手竞速和路线泛化。

当前状态：范围共识及 PRD 已整理；尚未开始代码实现、游戏接入或训练验证。七天是首轮探索窗口，按可运行里程碑推进。

## 当前文档

| 文档 | 用途 |
|---|---|
| [PRD](docs/PRD.md) | 当前目标、范围、功能与验收基线；实施入口 |
| [范围记录](docs/scope-decisions.md) | Q1–Q15 决策与用户补充要求 |
| [术语表](CONTEXT.md) | 统一领域词义 |
| [驾驶学习主方案](docs/driving-learning-design.md) | 基线、BC→SAC、自主学习循环与实施主线 |
| [奖励与有效性设计](docs/reward-and-validity-design.md) | 奖励投机防护、恢复分段、有效成绩及反例验证 |
| [倒带研究](docs/rewind-research.md) | 恢复能力的依据与待实测项 |

## 设计决定与研究

- [ADR 0001：首版使用已知道路信息与本车遥测](docs/adr/0001-known-route-first.md)
- [ADR 0002：将倒带恢复与正式驾驶分开](docs/adr/0002-separate-driving-and-recovery.md)
- [ADR 0003：成绩有效性先于性能排名](docs/adr/0003-validity-before-performance.md)
- [原始可行性评估](docs/feasibility-plan.zh-CN.md)、[参考项目审计](docs/reference-project-audit.md)、[FH5 接口研究](docs/fh5-environment-research.md)、[RL 方法研究](docs/rl-methods-research.md)

研究文档保留原始证据和历史候选；早期选车、实施顺序与排期建议以当前 PRD 和用户最新决定为准。
