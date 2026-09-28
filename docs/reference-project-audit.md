# FH6 VisionAI 参考项目审查

审查日期：2026-09-28。检查了 README、实际模型/训练/驾驶代码、配置、依赖、NOTICE，以及 18 页技术报告。未运行其模型、下载权重或训练数据；以下实验数字均为作者报告，未独立复现。链接指向审查时的 main 分支，后续可能变化。

## 方法与证据

这是 **行为克隆（BC）与多任务监督学习** 项目，未包含 RL 或 DAgger。作者报告三辆车在约 29 km 路程上完成闭环驾驶；但每种配置只有单次运行，完成由人工判断，测试路段约四分之三出现在训练录制中。因此不能据此推断 FH5 巨人赛的成功率、陌生赛道泛化或强化学习效率。训练数据需要申请，录制器未发布。[README](https://github.com/Ayin1412/ForzaHorizon6-VisionAI#-training)

实际网络采用 ResNet-18、几何预测头和独立控制 MLP。控制头读取 detach 后的路线偏移、航点、速度曲线和滑移预测，再结合当前速度输出转向和油门/制动。直接动作预测头用于训练表示，不用于实际执行。这一接口值得借鉴，但“几何接口有帮助”不等于所有辅助头都已被独立验证。[model.py](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/scripts/model.py)

输入配置是 320×180 RGB 三帧，间隔 0.1 秒；几何输出为 8–85 m 的九个航点。系统所谓 camera-only 指环境感知来源，不等于完全不使用遥测：配置包含速度、纵向加速度、上一油门状态。[meta.json](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/data/models/meta.json)

## 代码能够复用什么

实时链路为 Windows 窗口采集、UDP 遥测、PyTorch 推理、虚拟 Xbox 手柄。驾驶代码按精确长度接收遥测，默认目标控制频率为 30 Hz；其循环仅在未超时情况下补 sleep，不能保证实际达到 30 Hz。F9 接管、退出时手柄归零、轨迹叠加和仪表盘可作为 FH5 工程起点。观察模式实际需开启 overlay 才会在未接管时推理。代码未检查遥测或画面的过期时间，迁移时应补上断流归零机制。[drive.py](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/scripts/drive.py)

训练代码从轨迹位置和 yaw 构造车体坐标航点，从未来速度构造速度标签，再以加权 L1 监督。CSV 至少需要 session、frame_index、image_path、动作、速度、加速度、位置、yaw、路线偏移和四轮滑移；未知 session 会被拒绝。未来帧索引假设录制为 60 Hz。混合游戏 AI 和人工示范时，还存在随速度变化的转向单位映射。[train.py](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/scripts/train.py)

依赖为 torch/torchvision、NumPy、OpenCV、keyboard、vgamepad、windows-capture，训练另需 pandas，虚拟手柄需要独立驱动。依赖文件主要指定下限，没有提供锁定的可复现环境。[requirements.txt](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/requirements.txt)

## Pure pursuit 失败案例的适用范围

报告 §5.5、§7 明确说该手写对照只使用一组刻意简化的参数，没有做闭环调参或参数扫描。两种 pure pursuit 组合在第三个弯驶离道路；入弯前转向指令峰值仅 0.37–0.47，作者判断为控制增益不足导致转向不足。纵向控制换成速度曲线跟踪后仍失败，其该弯峰值约 30 m/s，即约 108 km/h，并非缓慢爬行测试。作者没有评估按弯道调度参数或 Stanley 控制。保留学习转向、仅替换纵向控制则完成路线。[技术报告，§5.5 与 §7](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/docs/Seeing%20Is%20Not%20Driving-Geometry%20as%20a%20Control%20Interface%20in%20a%20High-Speed%20Racing%20Game.pdf)

**工程判断：**这不能否定“已录制路线 + 遥测位置 + 自适应前视距离 + 保守速度规划”的 FH5 baseline。两者的路线来源、速度范围、控制标定与调参过程不同。但也不应承诺一个固定增益公式就能跑完全程。应先在弯道片段校准转向响应，检查误差随速度和曲率的变化，再逐步提速。

## 迁移到 FH5 的缺口

以下为根据源码接口得出的工作建议，不是作者已实现的能力：

1. **先验证 I/O。** 实测 FH5 遥测包大小、字段偏移、符号和单位；检查画面采集与手柄动作。窗口标题相近不代表协议和驾驶语义相同。
2. **重做数据采集。** 保存单调时钟时间戳、原始遥测、图像、实际命令和人工接管事件。不要让丢帧变成错误的未来轨迹标签。
3. **重新标定并训练。** 地图、视觉风格、相机、车辆和辅助驾驶设置变化会引入分布偏移；FH6 权重最多先作观察模式的迁移实验。转向标定、AI/人工单位映射都应在 FH5 重测。
4. **另建 RL 环境。** 当前发布代码没有 Gym 风格 step/reset、奖励、终止判据、重置状态机或经验回放。参考项目主要节省视觉 BC 与运行界面的摸索，不会直接解决 RL 实验管理。
5. **把完赛定义自动化。** 记录检查点顺序、完成进度、接管/回卷次数、离路时间与终点结果；不能用总里程代替巨人赛有效完赛。

## 硬件与复现边界

在所检查的 README、完整报告文本、配置与源码中，未找到具体 GPU 型号、显存峰值、单帧推理延迟或完整训练耗时。30 Hz 是运行设定，不能当成本机性能证明。应在 FH5 实际运行时测量采集至动作延迟、超时比例、GPU/显存占用，再决定视觉模型大小和训练批量。

**工程判断：**低维遥测策略与小型控制 MLP 值得优先尝试；视觉 BC 离线训练与游戏运行分时进行可避免资源争用。是否能跑动并不等于 RL 会在假期内收敛，游戏交互与自动重置效率需要单独评估。

## 许可证说明

源码采用 Apache-2.0；权重、标定数据、报告与图像采用作者标明的 CC BY-NC 4.0，并未赋予游戏画面本身的版权。个人 toy project 复用时保留相应署名和许可证；若后续公开衍生权重或数据，应分别核查适用条款。[NOTICE](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/NOTICE)

## 补充：交通车、AI 对手与路线范围

补查日期：2026-09-28。结论是：**可以在自由漫游或比赛中启动，但没有足够公开证据证明可靠避车、超车或在任意路线导航。** README 给出的启动场景包括自由漫游和比赛，设置表包含 Drivatar 难度；这些是运行说明，不是交通交互能力的测试结果，也不能据此认定评测时一定有其他车。[运行说明与设置](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/README.zh-CN.md)

对完整报告文本检索 traffic、opponent、overtake、collision、navigation、obstacle 等词，并复核方法、实验和局限性章节，未找到交通密度、对手数量、碰撞率、超车成功率或专门避障实验。报告的核心指标是路线完成、跟线、速度和滑移。因此其他车辆能否稳定处理属于**未证实**；资料也不足以确认训练或评测场景中完全没有车。[技术报告，§3–§7](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/docs/Seeing%20Is%20Not%20Driving-Geometry%20as%20a%20Control%20Interface%20in%20a%20High-Speed%20Racing%20Game.pdf)

源码没有专门的车辆检测/跟踪输出，也没有“跟车、超车、避让”决策模块；但是图像可以间接影响预测航点与速度，**不能从没有检测器推断它绝不会绕车或减速**。目前只能说没有可核验的可靠性证据。[模型接口](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/scripts/model.py)

路线也要分两层理解：驾驶循环根据当前画面实时预测局部路径，没有载入固定路线坐标表或回放预录动作；但接口中也没有目的地、左/右转导航指令或全局路径规划器。它不是硬编码的单赛道回放器，却也不是“给任意目的地就能到达”的导航系统。岔路处若缺乏画面中的赛车线等线索，仅凭其公开接口无法明确指定意图。[驾驶循环](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/scripts/drive.py)

公开测试是在指定路线进行；约四分之三测试路段已出现在训练录制中，作者将其定位为跨车辆测试，而非跨路线测试。因此不能把“在几辆未见车辆上完成测试路线”改写成“在任何新路线都能行驶”。[报告，§7 第 3–4 项](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/main/docs/Seeing%20Is%20Not%20Driving-Geometry%20as%20a%20Control%20Interface%20in%20a%20High-Speed%20Racing%20Game.pdf)

**对 FH5 的工程建议：**先将“固定巨人赛路线、无交通干扰的有效完赛”作为一个独立目标，再增加慢车跟随、单车避让与多车竞速测试。如果用户的目标包含正常比赛中的 AI 对手，应把车辆交互能力列为新增工作，而不是默认继承参考项目的演示效果。
