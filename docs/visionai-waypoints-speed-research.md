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
