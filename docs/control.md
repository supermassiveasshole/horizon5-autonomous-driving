# 低速控制校准（T02）

入口：`run_experiment(Control(config_file, output_dir), environment=...)`。本切片执行固定、限时的校准动作；不跟随路线，也不进行自主训练。

## 安装与运行

Windows 实机需要 [ViGEmBus 1.22.0](https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0) 驱动和可选 Python 依赖。驱动须单独安装；安装可能请求管理员权限或重启。ViGEmBus 已停止维护，当前机器兼容性以实测为准。校验官方安装程序的 Authenticode 签名，发布者应为 Nefarius Software Solutions e.U.。

```powershell
uv sync --locked --extra control
Copy-Item configs/control.example.json runs/calibration-config.json
uv run --locked --extra control fh5 control --config runs/calibration-config.json --output runs/calibration-001
```

最后一条仅检查配置，不创建手柄或发送输入。`vgamepad` 固定官方源码提交 `3f910aa8bbde49a576683db74ad5e4a0879f8a80`；uv 的构建变量关闭该库自动安装旧驱动的行为。请用 uv 安装本项目，不要另外 `pip install vgamepad`。

实机前准备：保持第六元素原车原调校，记录辅助与换挡设置，关闭自动转向和自动刹车；在可驾驶、平坦且有空间的路段停车。Data Out 为 `127.0.0.1:5300`。避免其他手柄、键盘或软件同时操纵车辆，正式校准期间不人工驾驶。

```powershell
uv run --locked --extra control fh5 control --config runs/calibration-config.json --output runs/calibration-001 --live
```

看到等待状态后切回 FH5。只有前台进程为 `ForzaHorizon5.exe`、收到活动遥测、车型/PI 匹配且低于起步速度时，才开始校准。示例约 7.2 秒；按 **F8** 或终端 **Ctrl+C** 解除输入。保持观察，停止后确认车辆响应。输出目录必须不存在。

## 动作、限制与停止

- `steer`、`longitudinal` 均为 `[-1, 1]`。纵向正值给油、负值刹车；不会同时踩两者。转向乘 32767、踏板乘 255，四舍五入为整数。游戏内左右方向与死区仍须实测。
- 示例限幅为转向 ±0.25、油门 0.15、刹车 0.5，最高 30 km/h，起步不超过 3 km/h。硬上限为 40 km/h、油门 0.25、转向 0.5、30 秒动作序列、60 秒等待，禁止无限试车。
- 20 Hz 调度采用实时时钟；断流、游戏时钟停滞、失焦、非活动状态、超速、车型变化、时间/位置跳变、接口异常均停止。退出等待后不自动重新接管。
- 独立线程每约 20 ms 检查 F8、焦点和 250 ms 命令租期；过期发送归零并锁止。它不能保证应对 Python 进程强杀、内核/驱动卡死或操作系统暂停，不能当作硬实时保证。
- “解除输入”指向手柄发送全零并断开虚拟设备，不代表车辆物理速度已经为零。异常时采用解除输入，正常动作序列包含刹车段。

## 记录与验收

除 `session.json`、原始 `packets.jsonl` 和报告外，还保存：

- `commands.jsonl`：原始策略目标、限幅后目标、成功下发值（失败为 null）、控制归属、调用起止时间及错误。
- `control.json`：版本、完整控制配置、结束原因、归零发送结果、独立线程事件。`sent` 仅表示驱动调用返回，不代表游戏已采用输入。
- 报告：命令周期、发送耗时，以及命令边沿后 500 ms 内遥测控制字段的首次变化。该“观察延迟”含接收/调度延迟，不能证明因果或代表完整车辆动力响应；人手操作会污染测量。无响应保留 null。

归零后继续被动记录约 300 ms；Ctrl+C/接口异常路径优先归零，可能没有这段观测。未完整结束的日志按不完整回放，不补造释放成功。回放不需要驱动。

实机先核对左右方向、油门与刹车映射，再用不同小幅度脉冲测死区和响应；分别演练 F8、失焦、暂停、停止 Data Out。每轮记录配置、原始数据哈希、观察和失败条件。保留原调校，不用改车掩盖控制问题。

在实机证据补齐之前，`game_response_validation=unverified`，`sustained_sampling_allowed=false`。通过合成测试、调用驱动成功、实际游戏响应验证是三项不同结论。此版本始终只开放短时校准；持续驾驶接口留待后续任务。
