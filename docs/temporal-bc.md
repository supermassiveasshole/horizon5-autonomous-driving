# 数值 Δt BC

对应 #35。`temporal-prepare` → `temporal-train` → `temporal-replay` 经实验入口完成离线准备、训练和冻结重载，不启动游戏或连接虚拟手柄。当前先使用历史示范；DXGI 原生时间与新采集分布仍需 #34/#37/#38。

## 准备与训练

```powershell
uv run --locked fh5 temporal-prepare --config configs/temporal-prepare.example.json --output runs/temporal-prepared
uv run --locked fh5 temporal-train --config runs/temporal-train.json --output runs/temporal-model
uv run --locked fh5 temporal-replay --model runs/temporal-model --dataset runs/temporal-prepared/dataset.json --report runs/temporal-reloaded.html
```

需要 `learning` 可选依赖。输出目录必须不存在。准备命令返回冻结 `dataset.json` 的 SHA-256；将它填入训练配置，数据路径相对该配置文件：

```json
{
  "version": 1,
  "dataset": "temporal-prepared/dataset.json",
  "dataset_sha256": "替换为准备命令返回的64位摘要",
  "seed": 20260930,
  "steps": 200,
  "batch_size": 32,
  "learning_rate": 0.001,
  "device": "cuda",
  "time_mode": "actual"
}
```

固定时间对照仅把 `time_mode` 改为 `fixed`，保留种子、步骤、批量、数据与结构。权重取固定最后一步，不依据留出误差选模型。CPU 使用 `device: cpu`。负面结果仍保存，不注入最低油门或修改预测幅度。

历史导入先从绑定原始记录重建并核对示范，独立核对前向片段边界。JPEG/PNG 只在这里解码，每个源帧只转换一次；训练和冻结推理读取数值像素。名义视图保留原选帧，其他历史模式从当时已交付、同片段的真实画面重选，保留原时间和不确定性；不复制帧伪造运动或修改源时钟。不可构成的模式与原始排除原因写入清单。

## 时间和模型契约

三帧由旧到新。相邻源时间先以整数纳秒做差，再换算成秒；在原本车状态、图像 age/mask、因果动作及参考特征后追加 `(Δt 秒, mask)`。完整历史才进行预测；尚未训练缺图推理。训练和 `FrozenNumericActor` 使用同一个特征函数，数值推理日志保存模型实际使用的完整特征。

`actual` 使用实测间隔和真实图像年龄；`fixed` 将**所有图像年龄和相邻间隔**替换为名义值，防止从年龄差推回实际间隔。遥测和动作历史年龄保留，因为图像重选不会修改这些信息。它们不包含图像源时间。原始时间仍在报告中展示，作为诊断依据。

模型版本 2 绑定像素/时间/动作契约、数据摘要、来源和尝试分组；旧 BC 不能静默加载为时间候选。旧模型继续要求 `--legacy-diagnostic`，新模型经同一数值接口加载。模型输入不包含全局时间、当前监督动作或未来轨迹。

训练强制核对保存前后预测，绝对容差为 `1e-6`；冻结报告的摘要写入模型清单。`temporal-replay` 重新检查完整数据与像素，逐决策核对帧、输入特征和预测；输入或预测漂移超过门槛会明确失败，不把旧的训练重载误差冒充本次回放结果。早期没有冻结核验记录的开发候选需要重新训练封存。

## 快照、报告与边界

`numeric-bc-snapshot-v1` 包含像素契约、动作契约、带证据身份的尝试组及数值决策。每项决策绑定帧元数据和像素哈希、同一示范的两种参考视图、独立监督及资格；同一帧不能被划到不同尝试组。训练消费精确摘要绑定的清单，修改数据必须生成新版本。后续原生采集可生成同一格式。

历史 `holdout` 已用于先前诊断，因此映射为 `development`，不称为未见过的最终评估；`evaluation` 缺失时报告空组。时间变体和两种参考视图留在原尝试组，不增加独立驾驶样本数。当前离线入口上限为 50,000 个决策、512 MiB 唯一帧引用；超限拒绝，不悄悄截断。长期分块数据治理仍由持续采集/数据集任务推进。

报告按决策展示数值像素的 PNG 副本、时间、真实标签、预测和动作历史消融；按名义/重选历史、参考条件、起步/左右转向/松油/制动及尝试组分层报告误差。复制最近动作作为诊断基线。mask 后误差变化和非零时间梯度只证明计算/依赖，不证明 Δt 带来驾驶收益。

页面播放/拖动需要独立交互核验。软件测试、历史数据回放、新 4K 动态采集和真实驾驶验收分别报告。旧 `bc-train` 与编码图像 `policy` 在线入口均已退役；历史模型仍可离线回放。数值候选进入 `realtime-drive` 仍须满足独立驾驶条件，不能据离线 loss 降低恢复实机驾驶。
