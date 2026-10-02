# SAC 训练诊断与恢复状态

`SACTrain` / `SACResume` 每完成一次有效更新，向输出目录的 `diagnostics/updates.jsonl`
写一行诊断，包含累计步号、实际抽样转移、来源角色、目标和损失。不在内存摘要中累计所有更新，
也不因日志变大而拒绝继续。使用标准文件缓冲，结束时 flush、fsync 并关闭。

`training-report.json` 和实验返回摘要的 `updates` 字段为：

```json
{
  "format": "sac-update-jsonl-v1",
  "path": "diagnostics/updates.jsonl",
  "status": "complete",
  "records": 4,
  "sha256": "<该文件原始字节的 SHA-256>",
  "error": null,
  "role": "optional_local_diagnostic; not required for checkpoint recovery"
}
```

`records` 是写入接口已接收的完整记录数；只有 `complete` 才表示该次输出成功刷新且有完整文件摘要。
读取时逐行解析，不需要将日志整体载入；先核验摘要，再将记录用于数值诊断。
示例的 4 是该次实际更新数，不是记录上限。

写入、刷新、关闭的 `OSError` / `MemoryError` 将状态设为 `unavailable`，摘要设为 `null`，
记录错误并停用该诊断流。可能存在部分文件或未刷新的记录，不能将其当作完整日志。
更新进度单独计数；诊断失败不回滚学习器、不伪造损失、不要求重新训练。
必要模型、经验或状态摘要的保存失败仍不能发布成功检查点。

日志是本次运行的可选诊断，不是恢复依赖。新检查点摘要记录它在创建时的状态，不保证移动或清理后仍存在。
续训、候选归档/恢复保留绑定的训练摘要和完整 learner；它们不会追踪、复制可选日志或祖先日志。
需要逐步诊断时另行保留这些文件。旧检查点 `updates` 数组保持原字节，仍可恢复。

SAC 单次 `steps` 是配置明确指定的非负更新预算，累计步号同样不设人为数值上限；
主动停止仍在完整更新边界保存。当前其他资源工作尚未完成：训练前后全量预测、预热明细、
经验索引和父循环历史仍待迁移；本片段不宣称整个训练进程内存已与数据量无关。
