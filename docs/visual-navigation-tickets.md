# 视觉导航任务索引

2026-09-29。用户已批准并发布：规格 [#30](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/30)，2 项新增、9 项修订。完整契约见[视觉导航规格](visual-navigation-spec.md)。

| 任务 | 本轮操作 | Blocked by |
|---|---|---|
| [T29：回放无历史参考的因果视觉导航观测（#31）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/31) | 新增 | [#26](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/26) |
| [T26：录制可信动作的同步短段示范（#27）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/27) | 修订 | [#3](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/3)、[#31](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/31) |
| [T27：训练并回放多模态 BC 初始化策略（#28）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/28) | 修订 | [#27](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/27) |
| [T07：回放练习并解释奖励与终止结算（#8）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/8) | 修订 | [#7](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/7) |
| [T08：让多模态 BC 策略实际驾驶短段（#9）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/9) | 修订 | [#7](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/7)、[#28](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/28) |
| [T09：重复评估冻结多模态策略并报告全部尝试（#10）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/10) | 修订 | [#4](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/4)、[#9](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/9) |
| [T10：完成多模态 SAC 采样、更新与驾驶对照（#11）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/11) | 修订 | [#8](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/8)、[#10](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/10) |
| [T12：中断后恢复多模态学习并检查数据兼容性（#13）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/13) | 修订 | [#11](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/11) |
| [T18：扩展全程覆盖并报告多模态策略推进（#19）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/19) | 修订 | [#9](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/9) |
| [T28：区分视觉接入、实际利用与驾驶收益（#29）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/29) | 修订 | [#10](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/10) |
| [T30：对照导航去向、进入状态与未训练路段（#32）](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/32) | 新增 | [#9](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/9) |

原生阻塞边与票正文逐项一致，合并依赖图无环。既有票的评论、标签、分配和状态未改；#1 父规格及 #26 原范围保留，已完成的 #2/#3/#23/#24 不重开。

## 执行顺序与边界

- #26 已补齐活动采集与冻结回放一致性证据，页面交互已获确认，见 [T25 验证](validation/t25-observations.md)。随后推进 #31 的可选参考输入，再接同步示范 #27 和离线 BC #28。
- 局部独立依据 #5→#7 与数据/离线训练并行；#9 真实短段后接 #4 支持的自动重复评估 #10，再接奖励 #8 与 SAC #11。
- 导航/迁移实验 #32 在 #9 后可独立推进，不阻塞 #10、#11、#19 或歌利亚正式全程验收。
- 自由漫游长期奖励、自动到达与无人值守不在本轮实现票中；先取得导航/到达依据，再另行切片。

发布绑定的规格提交为 `fb0361281a959c364ee74daab74ec6c1ad06c598`；后续仅更新文档中的已发布状态与索引。此次为文档与任务管理，没有启动实现、驾驶或训练。
