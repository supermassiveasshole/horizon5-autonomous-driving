# T25：因果多模态观测

对应 [GitHub #26](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/26)。本页记录观测 v1：输出可追溯的截图历史、本车状态和必选局部导航航点，不运行 actor、采集动作标签或发送游戏输入。同步示范另属 #27。#31 已另行实现可选/禁用参考及动作历史的 [v2 契约](navigation-observations.md)，本页 v1 命令与历史统计保持不变。

## 使用

```powershell
uv sync --locked --extra events
uv run --locked fh5 observe runs/vision-001 --config configs/observations.example.json --route runs/reference/route.json --report runs/observations-001/report.html
```

Python 入口为 `run_experiment(ObservationReplay(recording_dir, report_path, route_file, config_file))`。HTML 与同名 JSON 共用 `summary.observations`；清单保留文件引用及 SHA-256，可重新读取原始 RGB，不依赖旧 latent。

新被动采集可提前冻结配置和路线，并记录真实检查时刻：

```powershell
uv run --locked fh5 vision --config configs/goliath-fixed-v2.json --output runs/vision-observations-001 --camera chase-far --seconds 30 --observations configs/observations.example.json --route runs/reference/route.json
uv run --locked fh5 observe runs/vision-observations-001 --config runs/vision-observations-001/observation-config.json --route runs/vision-observations-001/observation-route/route.json --report runs/vision-observations-001/replayed.html
```

`vision` 的两个新增参数须一起给出。采集前复制经校验的路线、依据及配置；采集后通过同一构造器生成 `observations.html/json`。新时刻标为 `recorded_passive_checks_not_policy_calls`，还不是模型推理。旧录制按接收时间重建网格，标为 `reconstructed_receipt_grid_not_actual_policy_calls`；网格止于最后一个包，尾部静默仍看原采集报告。最多 100000 次检查，超限报错。

## 因果性与缺失

- 遥测取检查时刻之前最后一个已接收包，不使用图像的事后配对字段。接收时钟逆序或相等会拒绝观测构造，普通回放仍可检查异常。
- 每槽按 `decision_ns - history_offsets_ms` 选最后已交付的唯一帧。捕获、编码可用、交付时刻须有序；重复选择时优先保留最新槽，较旧槽缺失。不重复旧图填满历史；相邻检查可沿用同一帧，但其真实年龄继续增长。
- 年龄从捕获开始算，不从交付或存盘算。每槽年龄上限为槽偏移加 `max_image_age_ms`。示例采用 200/100/0 ms 三槽、新图上限 200 ms、遥测上限 100 ms。这些是数据完整性阈值，不是闭环可接受延迟。
- 暂停/恢复、断流、游戏计时异常、位置跳跃、定位连续性中断、失焦/恢复及明确的重开/倒带事件截断历史；边界发生后若还没有新遥测，旧车辆状态与航点也不可用。游戏时钟超过 250 ms 不推进也不可用。
- 缺图、陈旧、错误 RGB 元数据、坏像素、未知交付时刻及完整性错误均显式报告。纯遥测不算完整多模态输入。`usable` 不证明驾驶合法、安全或物理同步。

原始 UDP 不能识别所有倒带：既无可检测异常、也无外部事件的倒带不能据此排除，正式有效性仍需独立证据。

预处理版本 `source-rgb-v2` 保存原捕获 RGB uint8、尺寸/缩放方式，`normalization=none_rgb_uint8`；v2 明确历史去重优先保留最新槽。不拟合统计或新增训练缩放。更换缩放、归一化或历史策略须版本化。追尾远仅是相机档位，速度导致的外参变化保持 `dynamic_unknown`。

## 运动与路线

解析器 `fh5-dash-324-v2` 新增 float32：偏移 32/36/40 的车体速度 m/s、44/48/52 的车体角速度 rad/s、56 的 yaw rad。布局来源为 [go-forza-telemetry v1.2.0](https://github.com/csutorasa/go-forza-telemetry/blob/v1.2.0/v2.go)，本机核验见 [验证记录](validation/t25-observations.md)。车体系约定 X 右、Y 上、Z 前；pitch/roll 轴未独立实测，不视为完整姿态标定。非有限运动值或越界 yaw 产生 `motion=null`，保留其他有效遥测。

局部航点为水平航向坐标 `[右, 前]`，单位米：世界 X/Z 差为 `(dx,dz)` 时，右为 `cos(yaw)*dx - sin(yaw)*dz`，前为 `sin(yaw)*dx + cos(yaw)*dz`。它不是相机坐标或补偿俯仰/侧倾的完整 3D 变换。独立预期测试覆盖直行、左右航向和 ±π。

匹配沿用因果连续定位；远离参考线超过 10 m、回头路歧义和跳跃遮蔽航点。允许从中途唯一位置读取导航参考；参考尾部不拼接或外推。未核验走廊/检查点不妨碍读取先验，但参考轨迹不是道路宽度、唯一合法线路或奖励进度。

路线须由独立历史记录提前冻结。程序拒绝来源原始包哈希与当前整份记录相同，并绑定路线清单和资产哈希；不同哈希不能证明文件没有重叠。仍须按独立录制回合划分来源，禁止把本次或留出回合的未来轨迹改名作为导航输入。

## 报告

观测面板展示逐次历史图片、遮蔽原因、遥测包号和局部航点，支持播放、滑块及完整 JSON 检查。无效图片淡化用于诊断。来源区报告配置、覆盖率和年龄分布；原 RGB 回放区的事后配对不进入观测清单。

世界 X/Z 图以绿色实心点显示当前活动遥测位置，速度为零仍显示。非活动遥测的零坐标不作为位置；若游标之前有活动记录，用橙色空心点显示最后已知位置、包号及距当前游标的时间，并注明当前位置未知。向前回看至首个活动包之前不显示未来位置。这是回放提示，不补入策略观测，也不确认检查点进度。

软件测试从实验运行入口验证；实录验证单独报告。二者都不替代闭环实时性、动作标定或驾驶验收。
