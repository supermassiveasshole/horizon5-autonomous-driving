# 赛事生命周期（T03，开发中）

入口为 `run_experiment(EventRun(config_file, output_dir), event_environment=...)`；仍使用同一实验运行与回放边界。Windows 适配器与 `fh5 event` 已支持菜单自动操作，本机完成三次静止起跑核验及两次连续重开。当前只验收短时菜单探测，尚未开放持续采样或自主驾驶。

## Windows 菜单探测

沿用 T02 已安装的驱动，安装可选依赖：

```powershell
uv sync --locked --extra control --extra events
uv run --locked --extra control --extra events fh5 event --config runs/t03-menu-profile-v1/repeat-from-exit.json --output runs/event-001
```

默认只验证配置，不截图、不创建手柄、不建立输出目录。上述配置是本机忽略目录中的实测配置，模板与截图不随仓库分发；其他机器需按本节和下文校准自己的页面模板，不能直接套用合成测试夹具。

添加 `--live` 才开始运行。FH5 须处于前台和配置指定的起始菜单，Data Out 为 `127.0.0.1:5300`；窗口宽高比须与模板一致。本机为 2560×1440 客户区，缩放为 640×360 灰度。命令启动时就检查焦点；从终端启动需预留切回游戏的时间，例如先执行 `Start-Sleep -Seconds 5`，再执行带 `--live` 的命令。不可读取交互桌面的沙箱中会因失焦停止，应在实际 Windows 桌面会话运行，不关闭保护。

每次按钮脉冲为 80 ms，返回前释放；复用 T02 的 F8、失焦、250 ms 输入租期和独立线程保护。菜单适配器在零输入等待期也每 20 ms 检查并锁存停止或失焦，恢复焦点后不会自动续发按键。截图时间戳取采集前；慢截图不能伪装为新帧。仅采集前台 FH5 客户区，采集后再次核对窗口。截图依赖 [MSS](https://python-mss.readthedocs.io/stable/api.html) 与 [Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html)，离线回放不加载它们。

截图源由同一独立线程创建、采集与关闭：初始化最多等待 2 秒，单次采集最多等待 400 ms，关闭最多等待 100 ms。超时后停止实验，不复用该截图线程；底层原生调用若卡住，线程可能保留至进程退出，但它没有控制器引用，不能发送游戏输入。这些是调用方等待上限，不是操作系统或驱动的硬实时保证。

退出会归零并断开虚拟手柄；FH5 可能因此暂停并显示“控制器未连接”，可用键盘 Enter 关闭提示。`release_sent` 只表示归零调用返回，不证明车辆停稳。退出码 0 表示完成配置的尝试数且释放调用成功，尝试本身仍可能全部超时失败；非正常停止返回 4。

## 状态与证据

运行依次记录准备、起跑核验、尝试结束、重开和恢复核验。每次起跑都有独立 `attempt_id`。重开必须返回匹配车型/PI、低于 3 km/h、位于指定起点半径内的活动驾驶画面，才创建下一次尝试。超时结束记为失败；稳定的完成页面记为 `completion_observed`，仍不是经过路线、倒带和接管检查的有效成绩。

`EventEnvironment` 注入遥测、带单调时间的灰度画面、前台状态和停止信号；`pulse` 只支持有界的菜单按钮脉冲，返回前必须释放。外部适配器负责限制 `read/pulse/close` 的实际耗时及独立停止保护；状态机无法抢占卡死的驱动调用。当前没有驾驶动作输出。

画面必须尺寸正确、时间新鲜且递增。多个图块同时匹配才识别一个页面；同时匹配多个页面按未知处理。菜单动作要求连续两张匹配画面，每一步只发送一次；未出现下一状态则在阶段时限内退出。用户停止、失焦、断流、画面异常、运行中的车型变化或时间/位置跳变均中止。`IsRaceOn` 变化本身不算完成。

## 配置与回放

配置沿用版本 1 的车辆/调校/辅助快照，增加 `event_run`：

- `version`、`purpose`（默认 `event`）、`conditions_verified`、本地 `verification_evidence` 路径。
- `max_attempts`（1–10）、准备/重开时限（0.5–120 秒）、单次尝试时限（0.5–1800 秒）。
- `expected_car_ordinal`、`expected_pi`、`start_position_m` 和 `start_radius_m`。
- `screen_size`、`signatures`：每个页面包含 `box: [left, top, right, bottom]`、`template` 和 `max_error` 图块。
- `start_steps/restart_steps/finish_steps`：按顺序指定期待的 `screen` 与要发送的 `button`。

模板采用 `P5\n宽 高\n255\n` 后接灰度像素的 PGM。配置和证据在观察前固定并复制到本轮目录，原文件后续变化不影响本轮识别。普通 `event` 在条件未核验时只保存观测并退出，不发送菜单动作。

显式 `purpose: "restart_probe"` 允许在条件仍未核验时测试菜单：最多三次尝试、每次最多 10 秒、准备/重开各最多 30 秒；仍须提供本地证据及起跑、完成、重开模板和步骤。逐包检查活动遥测，速度超过 3 km/h 或出现非零驾驶输入立即退出，包括同批中间短暂出现的异常。不会把配置快照升级为已核验，也不发送驾驶动作。完整配置形状见 `tests/test_event_run.py`；其中合成图块不能用于真实游戏。

实测重开步骤为 `driving: START → pause_map: RIGHT → pause_exit: DOWN → pause_restart: A → confirm: A → ready: A`；成绩页则为 `finish: X → confirm: A → ready: A`。每步的页面名称对应独立模板；菜单布局或初始选中项不同会停止。`ready` 只证明页面类型，不能证明所选蓝图、辅助或调校正确，这些条件须另外核对。

除原始遥测外，保存 `event-config.json`、`event-assets/`、`frames/`、逐条刷盘的 `event-journal.jsonl` 与最终 `event-run.json`。回放核对原始遥测/画面/资产哈希、日志与最终摘要；缺失或损坏时从可读事件前缀恢复尝试列表，报告证据不完整，不补造释放成功。哈希用于发现损坏和不一致，不是对整个目录的防篡改签名。逐帧记录会占用较多磁盘，实机适配器须选择足以识别的低分辨率并测量吞吐。

`fh5 replay <run> --report <new-report.html>` 可离线查看尝试和菜单动作。`unattended_verified` 始终为 false；真实条件和剩余验收项见 [T03 验证记录](validation/t03-event.md)。
