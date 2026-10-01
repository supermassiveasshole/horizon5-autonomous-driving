# 学习记录容量清单

GitHub #16 提供只读容量规划入口 `LearningStoragePlan`，以及合成学习循环的阶段容量检查。输入已停止写入的学习会话，沿引用找到需要保留的文件。只读规划不启动游戏、模型训练或控制器；循环在开始下一阶段前使用同一计量规则。

## 使用

先停止会话及对其候选库的写入，再在仓库根目录执行：

```powershell
$session = Resolve-Path runs/learning-001
$stateHash = (Get-FileHash -LiteralPath "$session/state.json" -Algorithm SHA256).Hash.ToLowerInvariant()
@{
  version = 1
  root = (Resolve-Path runs).Path
  learning = @{ directory = $session.Path; state_sha256 = $stateHash }
  budget_bytes = 107374182400
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 runs/storage-request.json
uv run --locked fh5 learning-storage-plan --config runs/storage-request.json --output runs/storage-plan-001
```

示例的 100 GiB 是用户设定的容量参数，不是当前机器可持续运行的实测结论。配置和输出均使用新文件/目录。`root` 是允许访问依赖的路径范围，相对配置文件解析；`learning.directory` 相对该范围解析，也可为范围内的绝对路径。原始依赖若散落在范围之外，应先选择共同的合适根目录，不能跳过那些依赖以获得较小统计值。

成功时输出 `storage-plan.json`。退出码 0 表示保留文件总量不超过预算，4 表示已经超过；读取失败、过期状态、越界或缺少文件时不发布清单。CLI 仅打印概况，逐文件路径、大小和引用角色在 JSON 中。

## 保留范围

- 当前会话全部文件、绑定的录制/任务/奖励配置、原始路线及其标注证据、自动起跑所需模板与证据。
- 默认版本、探索版本、最新完整 learner，以及候选历史中的全部归档；同一实际路径只计一次，多种引用角色仍保留。
- 冻结评估、全部已开始尝试的遥测/动作/数值图像、起跑记录、独立审阅附件；未开始及失败结果不删除。
- 已封存采样绑定的原始文件、续训 replay 和经验帧、证据使用登记库。

计量是这些文件的逻辑字节数，既不是整个 `root` 的磁盘占用，也不是文件系统实际分配空间；不同路径的副本分别计算，硬链接不按物理块去重。不按扩展名推定 JPEG 可删：它可能已经成为独立证据。清单没有列出的文件也不意味着可以删除。

## 失败与边界

清单核对用于枚举依赖的清单摘要绑定（含主 replay 及来源 replay）、引用文件的存在性及读取期间已计量文件的大小/修改时间。它不重新加载 Torch 权重，也不替代模型完整性、驾驶有效性或可晋升性审核。目录读取失败直接报错，不静默跳过；引用中的目录链接和数据库链接在访问前拒绝。单次最多计量 100000 个文件。

依赖元数据读取预算为 256 MiB，包含嵌套清单、候选历史附件及最后的摘要复核；相同 checkpoint 的清单解析复用缓存。SQLite 读取保守地按每次访问的整个数据库大小预留预算。计量值写入 `metadata_read_bytes` / `metadata_budget_bytes`；它是预算收费量，不是系统磁盘 I/O 指标。最外层请求配置另有 1 MiB 上限，不占用依赖预算。

这是针对静止记录的保留依赖清单，尚不是并发写入时的事务快照、删除许可或完整磁盘配额管理器。只读报告明确记录 `files_deleted=0`、`cleanup_authorized=false`。

## 学习阶段容量检查

在[学习循环](learning-loop.md)配置中设置 `version: 2`，并增加以下必填项。版本 1 仍沿用原来的行为，不自动获得预算保护：

```json
"storage": {
  "root": "../runs",
  "budget_bytes": 107374182400,
  "min_free_bytes": 10737418240,
  "phase_reserve_bytes": 2147483648,
  "stop_reserve_bytes": 67108864
}
```

示例数值仅说明配置，不是推荐的实测容量。`root` 包含本次会话、版本库与全部依赖；相对配置文件解析。`budget_bytes` 限制保留依赖的逻辑字节数，`min_free_bytes` 是运行盘上的最低剩余空间，另外两项声明下一阶段及停止封存需要的余量。

初始化复制模型前、采样前、冻结评估前和版本归档前分别检查。仅当保留依赖加两项余量不超过逻辑预算，且磁盘剩余空间足以覆盖两项余量及最低剩余空间时，才进入下一阶段。检查发生在驾驶资源释放之后，不进入动作解除路径。

容量不足时返回 `storage_budget_exhausted`，释放后端并保存 `state.json` / `summary.json`。`storage_checks` 记录阶段、计量、剩余空间、声明余量和拒绝原因；不会删除原件或覆盖旧默认版本。容量恢复后，使用当前状态摘要调用 `LearningContinue`，再次检查并继续未完成阶段；已完成训练或评估不会因容量暂停而重新执行。配置已经冻结，不能在接续时悄悄提高预算。检查期间收到 `stop.request` 则按用户停止处理。

这里检查的是**阶段开始前的余量**，不会预分配磁盘块，也不能限制其他进程占用或保证阶段内写入不超过声明值。无法读取依赖或磁盘空间时按接口故障停止；真正磁盘写满仍可能妨碍最终状态落盘。初始化被拒绝前也会写入少量配置、锁和停止记录。阶段内容量监测、可验证的空间预留、展示数据降级、writer 长时停滞及真实 4K 游戏/训练共存压力仍属于 #16 后续工作。
