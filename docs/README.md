# 项目文档

当前操作按采集 → BC → 运行 → SAC → 评估推进。软件验证、真实设备验证和驾驶收益分别验收；当前实现状态及总入口见[项目 README](../README.md)。

| 阶段 | 先读 | 按需参考 |
|---|---|---|
| 采集 | [独立持续采集](guides/capture/continuous-collection.md) | [DXGI 与时钟](guides/capture/dxgi-capture.md)、[输入校准](guides/capture/demonstrations.md)、[遥测协议](guides/capture/recording.md) |
| BC | [数据选择与最终留出](guides/bc/collection-datasets.md)、[训练、调度与恢复](guides/bc/training.md) | [旧模型与历史示范迁移](archive/legacy-models.md) |
| 运行 | [只读检查与有界驾驶](guides/runtime/realtime-decisions.md) | [控制安装](guides/runtime/control.md)、[赛事菜单](guides/runtime/events.md)、[独立路线依据](guides/runtime/routes.md) |
| SAC | [初始化与离线更新](guides/sac/sac-learning.md)、[连续学习循环](guides/sac/learning-loop.md) | [采样与经验准备](guides/sac/sac-cycle.md)、[SAC 恢复](guides/sac/sac-resume.md)、[critic 恢复](guides/sac/critic-resume.md)、[经验混合](guides/sac/sac-mixture.md)、[临时模仿约束](guides/sac/sac-imitation.md) |
| 评估 | [冻结评估和全部尝试](guides/evaluation/evaluation.md)、[候选比较与保存](guides/evaluation/candidates.md) | [独立有效性](guides/evaluation/attempts.md)、[奖励结算](guides/evaluation/rewards.md) |

[学习存储与容量](guides/sac/learning-storage.md)和[训练诊断](guides/sac/training-diagnostics.md)用于保存、排查与接续。[传统跟随](guides/runtime/tracking.md)、[像素道路估计](guides/runtime/perception.md)、[恢复回放](guides/runtime/recovery.md)保留独立用途。已有编码图像还可进行[录制回放](guides/capture/vision-recording.md)、[因果观测回放](guides/capture/observations.md)和[可选参考导航回放](guides/capture/navigation-observations.md)。

- **当前设计**：[PRD](design/PRD.md)、[驾驶学习方案](design/driving-learning-design.md)、[模型策略](design/model-and-driving-strategy.md)、[奖励与有效性](design/reward-and-validity-design.md)、[多模态规格](design/multimodal-learning-spec.md)、[导航规格](design/visual-navigation-spec.md)、[采集方案](design/demonstration-collection-plan.md)、[资源策略](design/resource-policy.md)。
- **领域与工作约定**：[术语](../CONTEXT.md)、[ADR](adr/)、[代理工作约定](agents/)、[仓库约定](../AGENTS.md)。
- **证据与历史**：[逐次验证记录](validation/)保留当时结论；[研究资料](research/)提供依据；[范围决定](archive/scope-decisions.md)保留用户决策；[资源审计](validation/resource-limit-audit.md)汇总当前软件验证和剩余边界。任务当前状态以 [GitHub Issues](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues) 为准。

更新实现时维护对应操作指南；需求及验收以设计文档为准，历史研究与验证记录不自动成为当前能力声明。

代码组织与依赖约定见 [架构说明](architecture.md)。
