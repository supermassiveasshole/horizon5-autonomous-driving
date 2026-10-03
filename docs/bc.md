# 旧版 BC 模型回放与迁移

旧 `bc-train` / `BCTrain` 及其优化循环已退役。新训练统一使用[数值 Δt BC](temporal-bc.md)，采集期间使用[单配置后台训练](learning-schedule.md)。旧 `bc-replay` / `BCReplay` 继续读取 v1 权重和历史示范，检查契约并生成诊断，不连接游戏、不发送输入，也不修改旧模型。

## 运行

```powershell
uv sync --locked --extra learning
uv run --locked --extra learning fh5 bc-replay --model runs/bc-first --dataset runs/t27-dataset-reviewed/dataset.json --report runs/bc-reload.html --device cuda
```

回放使用新报告路径，保留旧模型、数据和源录制。检查使用 `uv run --locked --extra learning pytest tests/test_bc.py`，未安装 PyTorch 的基础环境会跳过模型测试。普通遥测录制/回放入口不导入 PyTorch。

## 迁移新训练

历史示范使用 `temporal-prepare` 导入，再用 `temporal-train` 训练新候选；新数值采集使用 `collection-bc-prepare` 准备，再用 `temporal-train` 或 `collection-bc-train`。旧 `bc-train` 调用只返回迁移说明，旧训练示例已移除，不静默套用旧配置。

这是数据导入后重新训练，不是旧权重转换或旧实验的精确重跑。目标时间历史至少需要两帧；只有单帧可用或不能满足因果时间契约的片段不能构成 Δt 训练输入。导入保留来源、时间不确定性和排除原因；旧 `holdout` 映射为已使用的 `development`，不冒充新的最终留出。历史来源模型也不能据此取得新采集或实机驾驶资格。旧单帧模型及失败观测的诊断仍由本页回放入口支持。

## 数据与输入契约

回放使用原始录制、绑定清单和嵌入的质量审阅重新导出样本，与提供的数据集逐项比较；不直接信任可编辑的 `bc_eligible`、历史动作或 split 字段。图片按哈希校验，保留按独立录制划分。原始文件、审阅及配置仍须保留；哈希核验不是对恶意共同重写所有文件的认证。

Actor 输入仅来自 v2 actor 字段：完整 RGB 历史、本车速度/车体速度/角速度、图像与遥测年龄、先前双轴动作及年龄/掩码、可选本车坐标参考航点及掩码。绝对位置、回合/目的地 ID、全局进度、标签、未来航点和质量审阅均不进入网络。完整图像历史或本车状态缺失时不预测，保留空值；已知失败但观测完整的片段可产生诊断预测，不能成为正确 BC 标签。

数值预处理使用固定物理尺度：速度与车体速度除以 100，角速度除以 5，毫秒年龄除以 1000，参考米数除以 100；掩码保留，缺失动作/航点填零且掩码为假。没有从留出数据拟合统计量。RGB 全幅双线性缩放后转 float32/255；不裁掉导航/小地图，也不假定追尾相机外参固定。

## 旧网络及历史训练方式

以下描述既有 v1 模型的生成方式，当前代码不再执行这条训练路径。

版本 `rgb-history-conv4-64-state64-fusion128-tanh2-v1`：每帧共享四层卷积（16/24/32/48 通道，步幅 2，ReLU），保留 3×5 空间池化特征，投影到 64 维；按历史顺序拼接，与 64 维数值分支融合，128 维隐层输出 tanh 双轴。编码器可训练，完整参数保存在 actor 内，无冻结 latent 缓存。采用 PyTorch 随机初始化，不下载外部预训练权重。

每个抽到的可信训练样本以有参考/无参考两种视图成对进入批次，实际占比各 50%；batch_size 必须为偶数。同一图像缓存 uint8，仅当前小批量转换浮点。Adam（默认 betas/eps，无 weight decay）、两轴等权 MSE；当前不实现未来轨迹辅助头，明确记录关闭、权重零。固定 seed、更新次数与最终一步权重，不使用留出选择超参数或检查点，不做早停。确定性算法不能承诺不同硬件/PyTorch 版本逐位一致。

动作含义保持 `xinput-lx-rt-lt-v1`：转向 [-1,1]；纵向正值为 RT，负值为 LT 服务刹车，非手刹。输出只是策略请求，尚不代表限幅、下发或游戏采用。

## 产物与解释

- `actor.pt`：actor 全部参数（包含 encoder）及绑定的架构/配置/输入/预处理元数据；只用 `weights_only=True` 加载，不序列化可执行模型对象。
- `model.json`：文件哈希、架构、来源、固定条件、训练配置、输入历史、掩码课程与训练诊断。元数据须与权重内绑定值一致，版本/形状/有限值不符时报错。
- `report.html/json`：全部样本的预测、标签、质量与原因；分别报告 train/holdout、有/无参考的双轴 MAE、RMSE、P90、录制/5 秒窗口/速度/转向/踏板分层，以及复制上一输入的诊断基线。空分层没有成绩。

历史图像梯度、编码器参数变化和固定数值输入下黑图干预只说明视觉参与计算/响应，不证明理解导航或带来驾驶收益。只有后续真实重复驾驶能回答这个问题。当前 SAC、冻结评估和实时驾驶使用数值时间模型；旧 v1 权重只能明确用于离线诊断，不包含合格 critic、优化器恢复状态或自动驾驶验收。

本轮实际结果见 [T27 验证](validation/t27-bc.md)。独立录制留出用于本轮初始化诊断；被查看后不能继续作为后续选型的未见最终评估。

实现采用 PyTorch 官方建议的 [state_dict 保存及 weights_only 加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，确定性限制参见 [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)。
