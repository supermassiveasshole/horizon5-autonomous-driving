# VisionAI 航点与速度预测核查

核查日期：2026-09-30。参考仓库固定为 `Ayin1412/ForzaHorizon6-VisionAI` 的 `be74b8def76e8c3247f91e310c5375db1e494756`（v18.3）。本文记录公开实现事实，不代表本项目已采用或实现同一架构。

## 预测什么

模型读取三帧 320×180 图像，帧间隔标称 0.1 秒。`wp` 预测沿行驶弧长 **8、12、17、23、30、40、52、66、85 米**的九个二维航点；坐标相对当前车辆，右为正、前为正。`wpspeed` 为每个航点预测一个速度，采用“当前归一化车速＋网络残差”。航点头在拼接车辆状态之前分出，不显式接收遥测车速；这不排除图像中含有速度线索。速度头明确读取车速。[模型定义与 forward](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L41-L85)、[发布配置](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/data/models/meta.json)

另外预测当前及 +0.5、+1.0 秒的循迹线偏移，和当前及 +0.25、+0.5 秒的前后轴滑移。它们与航点、速度一起构成控制头的中间输入。[头部布局](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/README.md#L71-L84)

## 标签从哪里来

训练脚本按每段录制的世界位置累积路程，在给定弧长处查找未来帧，再用当前朝向把未来位置变换到车体系；速度标签直接取**同一未来帧实际记录的车速**。跨位置跳变、倒行断点或不可信轨迹的航点监督会被屏蔽。因此，这些头学习的是示范中的未来行驶轨迹和速度分布，不能称为已求出的最优路线或车辆极限速度；部署时也不读取未来遥测。[标签构造](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L322-L359)、[部署输入](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L567-L583)

脚本另从未来速度标签由远及近回推制动允许速度，减速度参数为 6 m/s²；乘以 1.05 后，若当前速度过高，就把控制头的纵向动作监督改得更偏向制动。这是**训练阶段的标签修正**，与 `wpspeed` 本身的未来实际速度标签应区分。[参数与回推](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L109-L116)、[回推计算](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L361-L366)、[控制监督修正](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L517-L525)

## 如何驱动车辆

部署链路为：图像与状态 → 中间预测 → 学习得到的 `ctrl` 控制头 → 转向及有符号纵向动作 → 虚拟手柄。控制头只接收这些中间量和当前车速；中间预测通过 `detach()` 隔离控制损失的反向梯度，控制头不直接读取视觉特征。独立的直接 BC 转向／纵向头仍用于训练，但实际驾驶使用 `ctrl_steer` 和 `ctrl_throttle`；正纵向拆成油门，负纵向拆成刹车。[隔离及控制头](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L89-L109)、[执行动作](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L586-L635)

公开训练脚本使用示范动作、未来状态标签、加权 SmoothL1 损失及正则项，属于监督／模仿学习；这份实现不能作为“已通过 RL 自主超越示范者”的证据。以上是源码核查，未在本机运行其权重或复现驾驶效果。[监督目标与优化](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L616-L676)、[优化器及损失](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L779-L785)

## 五类监督信号的实现细节

### 航点与速度曲线

航点的标签采样使用累计路程上的 `searchsorted`，取达到目标距离的首个未来样本，没有在相邻位置间插值。位置除以 50 米、速度除以 100 m/s 后训练。任一远近航点缺失或跨断点时，该帧的 `wp_valid` 整体归零，九个位置和九个速度共同屏蔽，不是只屏蔽缺失的远端点。断点规则为单步位移大于 `max(3 × speed/60, 2m)`，或投影到前向的位移小于 −0.05 米；不能据此声称已识别所有倒带、暂停和重开。登记为 `wp_dirty` 的会话默认整段移除。[采样及屏蔽](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L322-L359)、[损失掩码](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L633-L640)、[整段移除](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L734-L743)

可把两类标签理解为：“看到当前画面时，示范者接下来沿路程走到了哪些相对位置、在那里开多快”。它们既不是官方道路边界，也不是未来地图坐标直接输入部署模型。速度头的残差结构使它容易表达未来减速／加速，但损失仍监督绝对的归一化未来速度，不直接监督最小圈速。[标签与输出](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L481-L487)、[残差速度头](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L78-L80)

### 循迹线偏移

部署解码器从游戏 Data Out 读取 `NormalizedDrivingLine`，类型为有符号字节，并除以 127；它不是从蓝色线条逐像素标注而来，也不提供未来循迹线的整条几何。训练脚本读取 CSV 的 `norm_line`，预测当前、+0.5 秒、+1.0 秒三个值，输出经过 `tanh`。这是带符号的无量纲量，不能转换成“偏离几米”；所核查源码没有明示符号左右语义，也未对解码值做夹紧。[字段及尺度](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L70-L79)、[读取](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L269-L277)、[模型输出](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L78)

未来标签按会话内 `shift(-30)`、`shift(-60)` 取得，假定原始数据 60 FPS；尾部不足则填当前值，不使用实际帧间时间或单独的未来有效性掩码。`session_clean` 的实际判据仅是该会话 `norm_line` 标准差至少 0.05，控制整组偏移损失；名称不代表每帧道路或动作质量都已审核。该逻辑与航点的断点分段没有联动。[偏移标签与掩码来源](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L307-L320)、[数据集掩码](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L478-L492)

### 轮胎滑移

部署实时字段明确是四轮 `TireCombinedSlip`，不是单独的 `TireSlipRatio` 或 `TireSlipAngle`。训练 CSV 的 `slip_fl/fr/rl/rr` 每轴取左右轮较大值，夹紧到 `[0,2]`，再除以 2；这一步**没有取绝对值**。输出顺序为 `[当前前轴, 当前后轴, +0.25秒前轴, +0.25秒后轴, +0.5秒前轴, +0.5秒后轴]`，未来样本按 15/30 行位移取得，末尾填当前值。预测头为线性输出，没有硬性的非负或上界约束。[实时字段](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L56-L79)、[滑移标签](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L368-L379)、[标签顺序](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L485-L492)、[滑移头](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L53-L55)

`slip_valid` 在派生列中全部设为 1，实际损失再乘 `steer_ok`，因此只有登记为**人类手柄**的会话监督滑移；Anna 数据即使已完成转向映射，仍不监督这一头。训练中分别以 50% 概率随机化上一纵向动作和纵向加速度，任一被随机化便取消该样本的滑移监督。没有对未来滑移使用航点的断点掩码。[人类标记](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L281-L305)、[滑移及增广掩码](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L489-L515)

滑移预测确实以动作为条件，但存在需要保留的训练／部署差异：训练**以及离线验证**向 `action` 传入当前示范动作 `target[:, :2]`；部署传入上一轮模型的转向与纵向指令。未来标签来自示范者随后真实驾驶，并非“固定当前动作执行半秒”的反事实实验。因此这只是动作条件预测头，不能称为已验证可供任意候选动作搜索的车辆动力学模型。[训练及验证调用](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L586-L631)、[部署上一动作](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L576-L609)

CSV 录制器未随仓库发布，本文核实到实时字段选择及训练如何读取 CSV；没有独立重建录制器对原始 `slip_*`、`norm_line` 或动作的全部前处理。[录制器未发布说明](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/README.md#L393-L401)

### 控制网络与动作标签

`ctrl` 接收 37 个标量：3 个偏移乘 3，18 个航点坐标，9 个速度，6 个滑移，以及当前车速。转向和纵向各用 `37→64→32→1` 的 MLP；转向另加 `line_to_steer` 线性残差，最后均用 `tanh` 限幅。中间量都先 `detach`，控制损失不回传到感知头；感知仍通过各自的监督及直接 BC 头训练。[控制头定义](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L56-L68)、[输入与梯度隔离](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/model.py#L89-L109)

转向监督取示范 CSV 的 `steer`；Anna 会先按当前速度插值得到比率 `ρ(v)`，把原始转向乘该比率并夹紧到 `[-1,1]`。Anna 的转向损失掩码相对权重为 0.4，映射发生饱和再乘 0.5；无映射时其转向监督屏蔽。这个**训练数据单位映射**不同于部署时按左右方向分别插值的 `SteerCalibration`，后者把希望游戏收到的转向变为摇杆命令。[Anna 映射](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L145-L163)、[监督权重](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L285-L305)、[部署校准](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L227-L249)

纵向示范动作 `u = accel - brake`，这是把两踏板压缩为一个有符号值，无法保留同时踩油门和刹车的两个独立幅度。直接 BC 头监督原始 `u`，`ctrl` 纵向头监督此前所述 `throttle_cf` 制动修正版。部署把正值变 RT、负值变 LT，并带有额外低速启发式：默认低于 25 m/s 时调整小油门／制动，低于 3 m/s 时取消负纵向输入。它们是执行脚本规则，不是速度头损失或学得的物理定律。[纵向原始标签](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L307-L308)、[修正标签](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L517-L525)、[默认执行参数](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L95-L100)、[踏板执行](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/drive.py#L597-L609)

## 损失、教师强制与限制

所有预测维度使用 `SmoothL1Loss`，先按各维有效样本的 mask 加权求均值，再按头部权重相加。偏移、航点、速度、滑移的头部总权重分别为 6、6、3、3，均摊到该头的输出维度；直接 BC 转向／纵向为 8／2，部署 `ctrl` 转向／纵向为 8／4。因此不能把这些数字直接解释为米、m/s 或物理重要性的倍数；标签尺度和有效样本分布也决定其作用。README 的“weighted L1”是概括，此处以源码中的 SmoothL1 为准。[权重](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L113-L116)、[维度分配](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L194-L204)、[mask 归一化](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L633-L647)、[SmoothL1](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L779-L785)

教师强制只在训练执行：当偏移、航点、滑移 mask **全部有效**时，以 50% 概率把整组预测中间量替换为真实标签，让 `ctrl` 学会使用准确几何；这不是全体样本一半都被替换。另对 `|当前偏移| > 0.25` 且偏移有效的样本，强制用真实偏移替换该组偏移输入，同时该类样本的控制转向损失相对权重乘 2.5。验证及实际部署不执行这种真值替换。[教师强制及转向加权](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L616-L647)

另外有两项手工正则，均不是 RL 奖励或硬性安全保证：

- **偏移符号正则**：对 `|真实偏移| > 0.15` 的有效值，鼓励 `预测值 × 真实符号 ≥ 0.2`，hinge 惩罚权重为 3，减少偏移方向弄反。[实现](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L649-L674)
- **曲率单调正则**：人工生成一对同车速、同滑移、零偏移的紧弯／缓弯圆弧，让紧弯的同向转向更大、纵向动作更小，margin 为 0.02，权重为 0.5。这是对 `ctrl` 的软先验，不是从轮胎模型算出的抓地力极限，也不保证真实驾驶满足该单调性。[参数](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L101-L102)、[合成几何与损失](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/scripts/train.py#L552-L581)

公开 README 明确说明没有 DAgger、没有 RL 微调；其结果用于支持这一模仿学习控制接口，不能替代本项目对于持续学习、奖励有效性、断点处理和实机可靠性的独立验证。[作者自述的限制](https://github.com/Ayin1412/ForzaHorizon6-VisionAI/blob/be74b8def76e8c3247f91e310c5375db1e494756/README.md#L411-L421)
