# T26 / #27：同步人工示范

对应 [GitHub #27](https://github.com/supermassiveasshole/horizon5-autonomous-driving/issues/27)。实验入口支持被动读取实体 XInput 手柄、RGB 和 Data Out，生成可追溯的双轴标签及整段训练/留出划分。不创建虚拟手柄、不发送控制命令、不训练策略。

## 采集与校准

在 Windows 上安装已有 `events` extra；XInput 通过标准库 ctypes 读取系统 DLL，无新增驱动。保持原车原调校、辅助设置和追尾远视角。先查看设备，再保存输入配置：

```powershell
uv run --locked fh5 input-devices
uv run --locked fh5 demonstrate --config configs/goliath-fixed-v2.json --output runs/demo-001 --camera chase-far --seconds 60 --observations configs/observations-navigation.example.json --route runs/reference/route.json --input-profile runs/input-profile.json
uv run --locked fh5 demonstration-replay runs/demo-001 --report runs/demo-review/report.html
```

`input-profile.json` 的 `version=1`、`mapping="xinput-lx-rt-lt-v1"`，`device` 复制设备命令输出中的完整对象；`calibration` 初始为 `{"status":"unverified","evidence":[]}`。采集参数 `--seconds` 包括等待切回游戏的时间；启动提示后再驾驶，F8 或输出目录 `STOP` 文件可结束。

首次在同一车辆/设置下依次左右转向、RT 起步、松开 RT 后 LT 刹停，记录用户确认、原始文件哈希和输入/遥测响应证据，另存 `verified` 配置。`evidence` 保存这些依据；可附 `conditions`（须与实验 snapshot 相同）。验证的是轴方向和踏板响应，不能据此声称完整车辆模型、死区逆映射或精确物理延迟已经标定。未标定录制可查看，不能产生有效 BC 样本。

微软 [XINPUT_GAMEPAD](https://learn.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_gamepad) 定义左摇杆有符号 16 位值、独立的 0–255 扳机值。映射为 `LX/32768`（负）或 `LX/32767`（非负），纵向为 `(RT-LT)/255`；仅在不同时踩踏板时有效。LT 是行车制动，手刹按钮不会映射成 LT。遥测 Steer/Accel/Brake 只作游戏响应对照。

[XInput 状态](https://learn.microsoft.com/en-us/windows/win32/api/xinput/nf-xinput-xinputgetstate) 按逻辑槽读取；[设备能力](https://learn.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_capabilities) 不是硬件序列号。启动时核对冻结能力，断连即停止，不自动换槽。无法保证识别两次轮询之间的同型设备热替换，也无法捕获所有驱动层输入合并；这仍需要固定设备和用户确认。

## 原始证据、时序和排除规则

- `vision.jsonl` 保存每次原始手柄轮询及开始/可用时刻。它们是主机时钟，不是真实按键跃迁时刻；游戏是否采用单次输入保持 `unverified`。报告并列显示邻近遥测，不把相关性当作因果延迟证明。
- 双踏板、任意按钮（包括手刹/倒带）、右摇杆观察镜头、键盘驾驶或其他 XInput 槽活动、失焦、断连及无效数值保留原始证据，但不生成双轴标签。轮询间隔超过 250 ms 时切断图像/动作历史；未知按键绑定不支持自动推断。
- `actions.jsonl` 只保存可映射输入，`action-history.json` 绑定动作与遥测哈希。当前监督动作取决策时刻之后的首次轮询；动作历史严格早于该时刻且已经可用。超时标签排除，不向过去倒填未来读数。
- `demonstration-session.json` 冻结遥测、画面索引、配置、相机记录、输入配置、动作及历史参考包；画面像素另由索引哈希验证。源变更失败而非静默重算。

## 整段划分与质量审阅

至少录制两个独立回合。每回合另存审阅 JSON，`packets_sha256`、`vision_sha256` 绑定原始证据：

```json
{"version":1,"packets_sha256":"<SHA256>","vision_sha256":"<SHA256>",
 "intervals":[{"start_ns":1000000000,"end_ns":2000000000,
 "quality":"trusted","evidence":"操作者确认并核对画面：正常驾驶，无暂停/倒带/碰撞"}],
 "intent_changes":[],"notes":"导航显示及目的地变更的核对依据；未知项明确记录"}
```

区间使用主机单调时钟，左闭右开、顺序排列且不重叠；质量为 `trusted/failed/unknown`，未覆盖处自动为 unknown。真实失误不能因按键合法就标为 trusted。意图变更用 `{"observed_ns":...,"evidence":"..."}` 记录；画面可见性仍沿用观测配置的 `unknown`，不得从有路线参考推断已看到导航。

数据集配置：

```json
{"version":1,"max_label_delay_ms":100,"future_offsets_ms":[200,500,1000],
 "runs":[{"directory":"demo-001","split":"train","review":"review-001.json"},
         {"directory":"demo-002","split":"holdout","review":"review-002.json"}]}
```

```powershell
uv run --locked fh5 demonstration-dataset --config runs/dataset-config.json --output runs/dataset-001
```

路径相对配置文件；源和输出分开保存。重复、重叠录制、条件/版本不一致、审阅哈希不符拒绝导出。当前双视图要求观测 v2、`reference_mode=optional`，历史参考须来自独立旧回合，不得来自数据集任何训练/留出录制。短段只能建立同路线留出，不能据此宣称跨路线泛化。

`dataset.json` 的 `sources` 保存来源/条件/审阅；每例 `views` 含实际屏蔽航点的 `no_reference` 及 `reference_assisted`。两者均只含 actor 契约字段，回合标识与世界位置不进入模型输入。图像路径相对各自源录制目录，训练器必须读取并校验真实 RGB。

`supervision` 独立保存动作、时间、有效掩码、排除原因及未来局部位置。未来目标只在同一连续片段的邻近遥测之间插值，使用决策当时位置/yaw 转换为右/前米；不跨暂停、倒带、重开、断流、控制权或意图变更、失败区间，不外推尾部。`bc_eligible` 仅表示通过这套数据检查，不代表示范最优、完赛有效或已取得学习收益。报告可逐段查看图像、原始输入、响应、未来标签和断点。
