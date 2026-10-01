# 局部尝试与独立有效性（T06 / #7）

`run_experiment(AttemptReplay(...))` 和 `fh5 attempt-review` 对**完整录制**作离线判定，不发送游戏输入，不要求自动重开。报告包含全部尝试、前向片段、排除区、局部起终点、控制归属、证据和正式全程结论。人工置位采用 `local-validity-v2`：包含导航事件、任务阶段边界及显式接口故障清单；自动起跑任务采用下述 v3。历史结果保留。

```powershell
uv run --locked fh5 attempt-review runs/my-recording --task runs/my-task.json --evidence runs/my-review.json --output runs/my-verdict
```

输出目录必须不存在。`attempts.json` 是结构化结论，`report.html` 展示完整遥测及逐次结论，`telemetry.html` 保留基础诊断，`route-snapshot.json` 保存解析后的几何与证据。退出码 0 表示全部局部尝试经人工证据核验通过；4 表示存在失败、违规、待核验或接口异常；配置/文件错误为 2。省略 `--evidence` 可以检查数据，但不能得到有效纪录。

## 冻结任务与证据

任务 JSON 字段如下；相对路径以任务文件目录为基准。

```json
{
  "version": 1,
  "task_id": "goliath-start-interior-v2",
  "scope": "local",
  "route_file": "route-v2/route.json",
  "route_sha256": "填写冻结路线清单的 SHA256",
  "geometry_source_sha256": [],
  "start_mode": "manual_placement",
  "control_owner": "calibration",
  "expected_car_ordinal": 2941,
  "expected_pi": 999,
  "max_speed_kmh": 20,
  "max_duration_s": 30,
  "no_progress_timeout_s": 10
}
```

当前支持 `human` / `calibration` / `policy` 控制归属；v1 任务仍为人工置位。v2 任务显式使用策略控制和 `automatic_event_ready`，冻结赛事配置及交接期限，详见[自动起跑](evaluation.md#自动局部起跑的证据)。此时采用 `local-validity-v3`，本入口留下 `automatic_start_unverified`，只有绑定准备与执行证据的批次审核才能消除该项；其他证据条件不变。

`geometry_source_sha256` 必须列出额外的走廊建模记录哈希；参考轨迹来源由路线包自动读取。来源复用会隔离结果。该清单仍依赖审核者如实申报，哈希不证明标注本身正确。未来实测前冻结任务；事后诊断不得冒充预先冻结的最终评测。

证据 JSON 必含 `version: 1`、整份 `packets.jsonl` 的 `recording_sha256`、`items`、`coverage`、`events`：

- `items`：`id`、`path`、`sha256`。文件必须在证据文件目录内；核对哈希后复制到结果的 `evidence/`。保留原始审核 JSON 和任务 JSON。
- `coverage`：闭区间 `packet_range: [首包, 末包]`、`checks`、`source`、`reviewer`、`evidence: [文件 id]`。五项覆盖为 `wall_riding`、`reset_boost`、`grass_shortcut`、`interventions`、`conditions`，必须覆盖整次尝试。不要把“没标事件”当成“已检查且正常”。
- `events`：`packet_index`、`kind`、`status`（`confirmed` / `suspected`），以及同样的来源、审核者和证据引用。来源为 `independent_review` 或 `policy_prediction`；后者不能确认有效或违规。
- 事件种类：持续蹭墙、复位提速、草地捷径分别为前三个检查名；另有 `incidental_contact`、`reasonable_cut`、`pause`、`rewind`、`restart`、`stop`、`human_takeover`、`human_placement`、`interface_fault`、`driving_failure`、`conditions_changed`、`race_start`、`game_finish`。
- 暂停、倒带、接口故障可带 `resume_packet_index`：恢复后的首包。中间区间排除；不提供恢复位置则排除到本次尝试末尾。倒带动画不算前向驾驶。
- v2 另接受 `navigation_recomputed`、`navigation_hidden`、`destination_changed`。前两者只记录，不证明到达；确认改目的地拆分片段并标记任务变化。已有任务不自动改绑，具体奖励与截断见[奖励说明](rewards.md)。

## 分段与结论

默认整份录制为一次完整尝试；经独立确认的重开从该包开始新尝试，之前的结果保留。原始包号不重排，坏包仍计入完整包范围；空记录保留 `[0, -1]` 的接口异常。暂停、倒带、停止、故障及解码器连续性边界拆开片段；不同片段的进度取最大值，绝不相加。当前不导出训练转移。

局部任务取每个连续片段第一次进入已核验起点的位置：参考里程 ≤0.25 m，沿用路线定位的 2 m 起点距离容差，并要求核验走廊、检查点成立。进入前为 `approach`，首次完整到达后为 `after_local_task`；完整记录仍保留。进入后不会通过重新挑选干净子段消除路径异常。该容差是局部启动定位规则，不是道路边界宽度。

| 结论 | 含义 |
|---|---|
| `valid_complete` | 连续局部任务完成，几何、条件、人工证据覆盖齐备 |
| `driving_failed` | 已观察到超速、超时、无进度期限、检查点失败或审核确认的驾驶失败/提前停止；其他疑点仍保留 |
| `invalid` | 独立确认违规、倒带、接管、车辆/条件不符；途中人工置位视为接管 |
| `pending_review` | 来源复用、证据缺失、疑似事件、起点或路径不在核验范围、活动/位置边界不明 |
| `interface_error` | 断流、坏包、捕获/控制/赛事日志损坏、释放证据不完整或持续超过 0.5 秒的游戏时钟冻结；不归为策略驾驶错误 |

多种原因并存时全部保留，主结论优先接口异常、确认违规、驾驶失败、待核验，最后才是完成。偶发接触和合理切弯只记录，不自动等同持续违规。瞬时加速度、IsRaceOn 切换、单次重复时间戳都不是碰撞、倒带或完赛真值。

`record_eligible` 仅代表**人工审核的局部纪录**；当前所有结果的 `automatic_promotion_allowed`、`unattended` 都为 false。正式全程仍要求 `full_race_start`、`ordered_official_checkpoints`、`game_finish_signal`、`whole_attempt_validity`，本模块不接受全程 scope。即使标注了起跑/完成页面，也不能把局部成功升级为歌利亚完赛。

## 识别范围

软件能执行独立事件判定、几何覆盖、时序分段和条件门槛；尚无经实机验证的自动蹭墙、草地或复位检测器。人工审核可支持局部结论，但仍可能漏检；未知项应保持未覆盖。完整自动评测批次另依赖已验证的自动重开。真实诊断及误报限制见 [T06 验证记录](validation/t06-attempts.md)。
