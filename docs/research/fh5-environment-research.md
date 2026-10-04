# FH5 环境接口与训练可行性核查

核查日期：2026-09-28。范围：Windows 上的 Forza Horizon 5，首版允许遥测和人工录制路线，先稳定完赛，再接入强化学习。本文是资料核查与实现建议，没有连接游戏、安装驱动或验证实机行为。

## 结论

基于游戏已有的 UDP 遥测，加上模拟 Xbox 手柄输入，可以搭建低维状态的路线跟踪环境。计算量很低，主要未知项是输入驱动兼容性、事件状态识别、恢复/重开流程以及数据和动作的时间对齐。公开 Data Out 文档描述的是单向输出接口；本次未找到官方同步 `step()`、任意状态 `reset()`、加速模拟或并行实例训练 API。不能把 Gymnasium 包装器等同于获得这些能力。[官方 Data Out 说明](https://forums.forza.net/t/forza-motorsport-7-data-out-feature-details/74013)

## 已核实的遥测依据

官方完整字段说明来自 FM7，不能直接把 FM7 的包偏移用于 FH5。两个独立开源解析器将 FH4/FH5 的格式判定为 324 字节；基础 sled 后有 12 字节间隔，Position 从 244 开始。实现应先保存真实原始包并校验长度，再选择解析布局，遇到未知长度明确报错。[Go 的 FH4/FH5 解析源码](https://github.com/csutorasa/go-forza-telemetry/blob/v1.2.0/v2.go)、[Rust 的格式判定和偏移表](https://github.com/0x20F/forza-telemetry/blob/master/src/decoder/formats.rs)

下表偏移来自上述 FH5 解析源码；单位和语义交叉参考官方字段表与 Go 数据模型。Yaw 弧度、Steer 范围等详细语义属于社区实现说明，需实机校准。[官方字段表](https://forums.forza.net/t/forza-motorsport-7-data-out-feature-details/74013)、[Go 数据模型](https://github.com/csutorasa/go-forza-telemetry/blob/v1.2.0/model.go)

| 字段 | FH5 偏移/类型 | 对本项目的意义 |
|---|---|---|
| IsRaceOn | 0 / int32 | 运行/暂停辅助标志，不能单独代表正在正式比赛或成功完赛 |
| TimestampMS | 4 / uint32 | 毫秒计时；需要处理回绕、重启和时间跳变 |
| VelocityX/Y/Z | 32/36/40 / float32 | 车体局部速度，轴为右、上、前；不是世界坐标速度 |
| AngularVelocityX/Y/Z | 44/48/52 / float32 | 局部角速度，Y 为 yaw rate；社区模型按 rad/s |
| Yaw/Pitch/Roll | 56/60/64 / float32 | 世界姿态；社区模型按 rad，须标定方向和零点 |
| PositionX/Y/Z | 244/248/252 / float32 | 世界位置，单位米，可录制路线并计算相对路径误差 |
| Speed | 256 / float32 | m/s，乘 3.6 才是 km/h |
| DistanceTraveled | 292 / float32 | 行驶距离，不是沿目标赛道的完成度 |
| BestLap/LastLap/CurrentLap | 296/300/304 / float32 | 圈速/本圈计时；不是路线进度百分比 |
| CurrentRaceTime | 308 / float32 | 比赛计时辅助信号 |
| LapNumber | 312 / uint16 | 圈号；需实测单圈长赛程结束时的变化 |
| RacePosition | 314 / uint8 | 排名；不是坐标或完成度 |
| Accel/Brake | 315/316 / uint8 | 0–255 控制量，记录时可归一化 |
| Steer | 320 / int8 | 社区说明为 -128–127 左至右；不是前轮转角（弧度） |

上述公开字段表与解析器没有提供检查点 ID、已通过检查点列表、官方路线几何、道路边界、碰撞对象或明确的比赛完成布尔值。`NormalizedDrivingLine` 与 `NormalizedAIBrakeDifference` 也不能当作完整道路地图；首版不应依赖其未经验证的语义。[FH5 解析器完整字段](https://github.com/csutorasa/go-forza-telemetry/blob/v1.2.0/v2.go)

官方早期说明标称 Data Out 为 60 fps；FH5 在本机的实际频率、暂停行为和 UDP 丢包情况需要测量。该旧文还有 localhost 限制，不能据此断言当前 Steam/Xbox PC 版一定不能向本机发包；本机回环、本机局域网 IP 的可用性属于首日验证事项。[官方说明](https://forums.forza.net/t/forza-motorsport-7-data-out-feature-details/74013)

## 路线、奖励与完赛判据：建议设计

以下为工程设计建议，尚非游戏能力实测结论。

1. 人工完整开一遍目标比赛，保存 `(timestamp, position, yaw, speed, controls)`；同时录屏，标注检查点、事故和赛程起终点。将有效行驶轨迹平滑、按弧长重采样，再生成带顺序的参考路径和保守目标速度。手工开的轨迹是可行路线，不天然是最优赛车线，也不应直接当道路中心线或道路边界。
2. 用世界位置投影到上次索引附近的参考线段，求弧长 `s`、横向误差、航向误差和前方曲率。考虑高度区分立交，限制单位时间允许的索引跳跃，避免交叉、相邻道路和绕圈误匹配。
3. 奖励使用合法连续路径段上的 `Δs`，并惩罚偏线、停滞、倒行和控制抖动；拒绝传送、回溯、漏检查点后的跳变。不要直接奖励 `Speed` 或 `DistanceTraveled`，否则原地绕圈、沿错误道路高速行驶也可能拿正奖励。
4. 轨迹进度是内部估计，不能证明游戏承认通过了检查点。检查点附近增加目标约束，HUD 检测“错过检查点”等异常；必要时人工标注检查点区域。最终以游戏结果界面/完赛确认作为成功标签，配合路线顺序与圈数验证。首个里程碑建议定义为完整目标赛程、无人工接管、无回溯、游戏确认完赛；不要求获胜。
5. 起步用 Pure Pursuit 或 Stanley 转向，加速度规划和 PID 速度控制。由于输入是手柄轴而非轮角，需要实验标定死区、非线性与速度相关响应；算法公式的车辆几何参数不能不经校准直接套用。

## 控制与画面采集

`vgamepad` 的 `VX360Gamepad` 支持左摇杆浮点输入 [-1, 1]、扳机 [0, 1]，通过 `update()` 下发；其 `reset()` 只清空手柄状态，不会重置游戏。可以映射转向、油门和刹车，并用按钮操作重开菜单。[vgamepad API](https://github.com/yannbouteiller/vgamepad)

Windows 后端依赖 ViGEmBus。其仓库于 2023-11-02 归档；维护者明确说明停止更新，但归档本身不使现有安装失效。因此是可试验方案，不是当前 Windows/FH5 兼容性的保证。先做小范围输入烟测并记录驱动版本；必要的安装是独立步骤。[ViGEmBus 仓库](https://github.com/nefarius/ViGEmBus)、[维护者 EOL 声明](https://docs.nefarius.at/projects/ViGEm/End-of-Life/)

DXcam 支持 Windows Desktop Duplication / Windows Graphics Capture 后端，适合后续获取游戏画面和 HUD。首版无需把全屏像素输入策略，画面可先用于事件状态识别和诊断。实测应关注新鲜帧、HUD 区域、窗口切换和采集延迟，不能把库的宣传帧率当作 FH5 的闭环性能。[DXcam 作者文档](https://github.com/ra1nty/DXcam/blob/main/README.md)

## 训练环境边界与待验证项

EventLab 2.0 官方说明可设置创建地图时的天气与时间，方便做条件受控的测试场景。[官方 EventLab 介绍](https://forza.net/news/forza-horizon-5-horizon-creatives) 但本项目仍需在用户已安装版本中检查目标赛程/蓝图能否稳定使用固定天气、固定车辆调校、单人无干扰、固定出生点与重开流程。不能由 EventLab 存在推出任意赛段重置或读取内部规则状态的程序接口。

| 首轮验证 | 通过标准/需要记录 |
|---|---|
| UDP 连通与格式 | 在手动正常驾驶时连续录制至少数分钟；确认包长、频率、时间戳、数值和丢包 |
| 坐标和朝向 | 低速直行、左右转、坡道；确认世界竖直轴、yaw 零点/符号/回绕、Steer 符号 |
| 连续控制 | 左右小输入、油门/刹车分别测试，实测动作到遥测响应延迟，检查其他手柄是否抢控制 |
| 状态识别 | 记录自由驾驶、赛前倒计时、正常比赛、暂停、漏点、回溯、重开和结果页的遥测/画面 |
| 自动重开 | 从几种失败状态连续重开，验证恢复到同一起点的可靠性和耗时；超时必须释放输入并停机 |
| 失联与人工接管 | UDP 过期、画面停滞、程序异常/失焦时停止继续给油；保留紧急中断热键 |
| 一致性 | 固定车辆、调校、辅助驾驶设置、天气和画质帧率；不同配置单独记录实验标识 |

初期通过菜单重开整个事件是最容易定义的 episode reset 候选；回溯用于调试/数据恢复时要切断轨迹与 replay buffer 连续性，不能当作普通前向 transition。训练不能假设能从长赛道任意位置瞬移重开，课程学习优先使用实际可创建和稳定重开的短路线。

对于“环岛赛程”，当前仅按本体 Goliath/歌利亚赛事理解；本次未找到可用于严格引证的官方里程数字。实际建图时用录制轨迹累计弧长估计长度，并以用户游戏中的赛事名称/地图确认，避免和其他长赛程或 DLC 的 Goliath 混淆。
