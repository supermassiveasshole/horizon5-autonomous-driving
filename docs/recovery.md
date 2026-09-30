# 恢复监督回放（T11 / #12，开发中）

已实现合成任务/UI 信号上的有界规则监督器，可复现释放、倒带请求、恢复确认、历史重建和新片段许可的顺序。**当前没有实机恢复适配器，不连接手柄，不调用已有菜单重开，不执行驾驶或清空真实模型状态。** #12 保持开放。软件接口为 `run_experiment(RecoveryReplay(config_file, trace_file, output_dir))`，沿用已确认的实验运行边界。

```powershell
uv run --locked fh5 recovery-replay --config configs/recovery-synthetic.example.json --trace configs/recovery-trace-synthetic.example.json --output runs/recovery-example
```

输出目录必须不存在。保存原配置、完整原信号、两个 SHA-256、`recovery.json` 和静态 `report.html`；从副本重放得到相同决策。退出码 0 仅表示合成回放完成，包含按预期失败或停止的案例；输入无效返回 2。没有 `--live` 参数。

## 信号与状态

`source_kind` 只接受 `synthetic`。`at_s` 为合成环境的单调墙钟；`observed_s` 是信号采集时刻。它们不是本机运行该命令的 CPU 耗时。缺信号时用 `tick` 推进时钟，截止点优先于迟到的确认。配置是工程候选，不能作为 FH5 倒带可用性或时限实测。

| 阶段 | 必要确认 | 下一条意图 |
| --- | --- | --- |
| 正常片段发生确认失败 | 独立任务管理的 `failure` | 关闭失败片段，`release` |
| 等待释放 | 与请求号一致的 `released` | `rewind` |
| 等待倒带 UI | 新鲜的 `rewind_ready`，请求号一致 | `resume` |
| 等待返回驾驶画面 | 新鲜的 `resumed`，请求号一致 | 增加 generation，`reset_history` |
| 重建历史 | 对应 generation 的 `history_ready`；足量新鲜、中性输入、条件/定位可用的正向样本 | 新片段与 `allow_driving` |
| 任一阶段无法继续 | 超时、限次、用户停止、接口错误或记录结束 | `release` 并停止 |

`reset_history` 列出时间、路线定位、动作、图像、循环网络及控制器状态的清除要求，带 `discard_before_s`。`allow_driving` 带新 generation 和新的起始包号；旧片段不提供前驱动作。真实运行时以后必须实际执行这些操作并提供确认，不能仅从此报告读到意图就声称已清空状态。

`sample` 中的 `conditions_valid` 和 `route_valid` 是独立任务管理输入，分别表示条件及路线定位可用；`neutral` 表示用于恢复核验的输入已归零。这些字段不是本模块从画面/遥测识别的结果。`failure` 接受 `off_road`、`missed_checkpoint`、`unrecoverable_heading`、`stalled`。后三种驾驶失败判定同样不能由人工填入合成信号来验收实机检测精度。

## 边界、限次和结果

- 同一物理时钟样本不充当新的前向样本；倒带逆时间、动画及等待期不进入片段。恢复后重新累计历史，失败片段和 `parent_fragment_id` 保留。此处只输出包归属，不导出 SAC transition 或奖励；已有[独立有效性](attempts.md)与[奖励结算](rewards.md)仍负责真实记录的判断。
- 每个阶段、整次会话、总倒带次数均有上限。正常驾驶只有新的合法进度前沿累计增加至少 1 cm 才刷新停滞计时；往返不能刷新。恢复后重置局部停滞锚点，但不会重置总次数/总耗时预算。
- 驾驶期观测过期先截断并停止；不能把收不到数据归因于驾驶停滞。旧请求确认、过期 UI、错误 generation 均不能开启新片段。恢复不可用或次数耗尽时保存停止，目前不假设任意场景的重开已验证。
- `recovery_success_fraction` 分母是全部恢复请求，含能力不可用和次数耗尽；另列实际产生的倒带意图数、成功数和恢复墙钟耗时。该耗时来自合成时钟，不是 FH5 性能指标。
- `finish` 只表示外部提供到达信号。失败后到达仍为失败尝试，`no_rewind_completion=false`。未失败的合成运行也不升级正式成绩；不拼接片段进度或奖励。

## 剩余验收

需要在蓝图 `105657219` 核对倒带是否可用、按钮绑定、页面状态、恢复点与计时/检查点语义，并接入真实输入释放、历史重建及恢复后前向驾驶。还需核实失败退出或已验证重开、成功率和真实耗时。当前没有这些新实机证据，不能宣称程序已经会在游戏中自主倒带。夜间实现不启动 FH5 或 Steam。
