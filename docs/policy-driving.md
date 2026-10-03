# 旧模型驾驶入口迁移

旧 `policy` 在线入口及 `PolicyDrive` 已退役；调用 `fh5 policy` 只返回迁移错误，不加载模型或连接设备。当前模型驾驶统一使用 [`fh5 realtime-drive`](realtime-decisions.md#有界驾驶命令与条件绑定)。

旧 v1 BC 模型和 `policy` JSON 不会自动转发为新配置。新入口使用来自连续数值采集、符合当前运行条件的[数值 Δt BC（temporal BC）模型](temporal-bc.md)，按 [`configs/realtime-drive.example.json`](../configs/realtime-drive.example.json) 填写采集配置、候选模型、独立任务和只读运行依据。默认只校验；准备好所需资产并满足条件后，显式 `--live` 才连接控制。具体条件见数值驾驶说明。

当前数值驾驶使用无参考策略，独立任务路线只用于起点、范围和结束核验，不作为策略航点输入；它不等价支持旧 `required` / `optional` 参考模式。停止时解除输入，不保证车辆已经刹停，也不沿用旧路径的终点制动流程。实机驾驶仍待验收。

已保存的旧录制继续支持离线读取，无需游戏，也不会重新发送输入：

```powershell
uv run --locked fh5 replay runs/policy-001 --report runs/policy-001/replay.html
```

使用新的报告文件名。旧录制、模型与[历史验收记录](validation/t08-policy-driving.md)保留原结论，不因入口退役或回放成功而成为新路径的驾驶证据。
