# T29：可选参考的视觉导航观测

对应 [GitHub #31](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/31)。在同一 `run_experiment` 入口构造观测 v2，保留 [v1](observations.md) 的必选参考契约及历史统计。本切片只回放输入、记录来源，不执行策略、采集真实动作或发送游戏输入。

## 使用与参考模式

```powershell
uv run --locked fh5 observe runs/vision-001 --config configs/observations-navigation.example.json --report runs/navigation-001/report.html
uv run --locked fh5 observe runs/vision-001 --config configs/observations-navigation.example.json --route runs/reference/route.json --evaluation-route runs/survey/route.json --report runs/navigation-002/report.html
```

每次使用新报告文件名。Python 使用 `ObservationReplay(recording_dir, report_path, route_file, config_file, evaluation_route_file=None)`；无参考时传入 `route_file=None`。

| `reference_mode` | 行为 |
|---|---|
| `required` | 缺失或坏资产报错；无法可靠定位或航点不足时观测不可用 |
| `optional` | 正常缺省为 `absent`；坏资产为 `invalid_asset`，保留错误并隔离；两者都屏蔽航点，不单凭参考缺失使有效 RGB/本车状态失效 |
| `disabled` | 不打开、校验或缓存 `--route` 指定的文件；所有航点缺失 |

v1 缺失参考仍报错，不接受 `--evaluation-route`。v2 参考与当前录制必须来自不同历史数据；同一录制的未来轨迹不能成为自身导航输入。错路导致远离参考时报告 `outside_reference`，邻近往返道路报告 `ambiguous`，定位跳变报告 `discontinuity`；不能仅凭局部几何识别所有“同一位置但目的地不同”的错路。路线匹配不是任务意图识别器。

新被动采集可只传 `--observations configs/observations-navigation.example.json`，省略 `--route`；配置和实际检查时刻仍冻结。若主动提供的资产无法冻结，采集前直接报错并关闭环境；离线回放则展示资产隔离状态。禁用模式不会复制参考。相机仍记录实际采集档位，动态外参保持未知。

## 策略输入与独立证据

JSON 的 `summary.observations.decisions[].actor` 是后续模型的数据视图：

- RGB 文件引用、像素哈希、图像掩码与真实年龄；文件引用仅用于加载像素，路径和哈希不能编码成模型特征。
- 速度、车体速度和角速度及状态掩码/年龄。绝对世界坐标、绝对航向、全局里程、回合/目的地/包号和单调时钟均留在诊断字段，不进入此视图。
- 本车航向坐标中的可选局部航点与掩码。
- `[转向, 有符号纵向动作]` 历史、掩码与实际年龄。

后续训练不得将整条 decision 或报告元数据直接作为模型输入。`usable` 表示图像/状态合格且符合声明的参考条件；动作缺失可通过掩码表达，不代表已有模型支持此输入。`policy_support=not_evaluated` 与 `task_assessment.scorable=false` 独立保留。

`--evaluation-route` 只在独立样本副本上生成路线诊断；其检查点、走廊、定位状态与进度不会进入 actor，也不会切断 actor 历史。没有完整任务判定器，已有参考证据也仅标为 `reference_evidence_only`，不会宣布合法完赛或自动晋升。

`navigation` 显式保存 `display`（`unknown/full/braking_only/off`）、`visibility`（`unknown/human_reviewed_visible/human_reviewed_occluded`）及 `evidence` 文字列表。人工可见性判断需要依据，但不作为在线识别器或模型输入。示例默认未知，应按录制实际条件填写；不解析或宣称 `NormalizedDrivingLine` 是未来路径。

## 因果动作历史格式

旧录制没有动作源时保持 `absent`，历史值为 `null`，掩码为 false；不用 UDP `Steer/Accel/Brake` 补标签。真实动作源、标定与示范采集留给 #27。

导入受控记录需在录制目录提供 `actions.jsonl` 及 `action-history.json`。后者包含 `version: 1`、`mapping_version: "dual-axis-v1"`、`actions_sha256`、`packets_sha256`；哈希绑定原始文件字节。每个 JSONL 对象恰好包含：

```json
{"occurred_ns": 1050000000, "available_ns": 1060000000, "telemetry_segment": 0, "source": "human_input", "steer": 0.2, "longitudinal": 0.4}
```

时间使用采集端同一单调时钟，`occurred_ns <= available_ns`，动作发生时刻不能重复。来源只接受 `human_input/controller_sent`；转向与纵向值均为有限的 `[-1,1]`，正纵向为油门、负值为减速刹车。来源声明和哈希验证不等于实机标定证据。

每槽选择在 `decision_ns - action_history_offsets_ms` **严格之前**已知的最近发生动作，排除等时当前标签；迟到的旧动作不覆盖已知新动作。年龄从动作发生算，上限为槽偏移加 `max_action_age_ms`。同一先前动作可延续到多个槽，但不会刷新年龄。暂停、恢复、倒带、重开、断流、失焦和遥测片段改变截断历史；缺失、陈旧及跨片段均屏蔽。未知或损坏动作文件整体隔离，错误单列；最多导入 100000 条动作。

图像仍按真实交付时间和唯一帧选择；保留 v1 的年龄、缺失、完整性和时钟校验。真实录制对照与局限见 [T29 验证](../../validation/t29-navigation-observations.md)。
