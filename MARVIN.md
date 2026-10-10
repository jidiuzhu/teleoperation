# MARVIN M6-CCS 4.0 左臂基础遥操作

本路径使用控制器 **100341**、A 通道、第一代 Wuji **左手**、左手套，以及 Windows SteamVR 的 Tracker 输入。服务端默认只读，不使能手部或下发机械臂运动；不再执行原来的 RM75 HOME 动作。传入 `--enable-motion` 后还需在客户端按 T 才启用。现有 RM75 后端保留，启动同样不再自动回 HOME。

当前范围是无 Kinect 的基础遥操作。MARVIN/左手的 Kinect 距离计算、手眼标定、ACTBH 部署和左手 VR 输入不在本次支持范围；相应组合会提前报错。27 维录制可以检查遥操作，但不是原 RM75 策略的兼容数据。

## 1. Ubuntu 环境与离线检查

在仓库根目录、专用 Python 环境中安装 `requirements-marvin.txt`。其中手套和机械手的官方 wheel 必须匹配 Python 版本及 ARM64 架构；如果 pip 找不到对应 wheel，应取得厂家提供的兼容包。安装 Ubuntu 版本本身不能证明 ARM64 支持。该路径不依赖 RealMan SDK、Pinocchio、CasADi 或 Ubuntu 上的 OpenVR。

```bash
python -m pip install -r requirements-marvin.txt
python teleop/scripts/build_marvin_sdk.py
python teleop/scripts/check_marvin.py
```

`third_party/marvin/UPSTREAM.md` 记录固定的厂家源码提交。构建脚本只编译 `contrlSDK` 和 `kinematicsSDK`，不会连接或升级机器人。生成的 `.so` 不纳入 Git；换电脑需重建。适配器在使用原生库前检查 Python/C++ 结构体大小及关键字段偏移。厂家其他示例中的自动清错和双臂动作没有移植。

离线检查输出库布局检查和 FK→IK 误差。之后可单独读取机械臂，无需接手或手套：

```bash
python teleop/scripts/check_marvin.py --arm-ip ROBOT_IP --side A --duration 10
```

`ROBOT_IP` 是控制柜地址，不是 Ubuntu 主机地址。程序只读取版本、关节反馈并计算法兰位姿；控制器必须报告 100341 系列。核对这些关节角与厂家上位机一致。

## 2. Windows Tracker 发送端

用本次更新后的 `teleop/scripts/send_tracker_windows.py` 替换 Windows 的旧文件。保持 SteamVR 运行：

```powershell
py .\send_tracker_windows.py --host 10.214.203.3 --serial LHR-C87FB5D3
```

使用 UDP 6210，发送版本 2 的消息。Ubuntu 会发回短随机校验消息，Windows 在新位姿包中回传它；接收端根据本地发送时间限制消息年龄，无需比较两台电脑的时钟。UDP 两个方向都必须可达。旧版本 1 只支持诊断，不允许作为实时遥操作输入。协议不提供身份认证，只适用于可信网络或 VPN。

原 `receive_tracker_udp.py` 仍可诊断版本 1/2 消息，但其 `TRACKING` 只代表报文状态，并非上述往返时效验证。正式客户端启动前退出诊断接收端，避免争用 UDP 6210。

## 3. 启动只读服务端

在 Ubuntu 仓库根目录：

```bash
python teleop/scripts/teleop_real_server_wuji.py \
  --arm-backend marvin --arm-ip ROBOT_IP --arm-side A \
  --hand-side left --hand-device YOUR_HAND_USB_SERIAL \
  --camera-mode none --flange-thickness 0
```

默认模型是 `third_party/marvin/CommonConfig/config/ccs_m6_40.MvKDCfg`，左臂选择 `arm_type=0`。模型中的毫米/度在适配器边界转换为项目的米/弧度。

## 4. Ubuntu 客户端：手套 + 网络 Tracker

将左手套网络配置好。在第二个终端运行：

```bash
python teleop/scripts/teleop_real_client_wuji.py \
  --devise glove --hand-side left --camera-mode none \
  --glove-sn YOUR_GLOVE_SN --tracker-serial LHR-C87FB5D3 \
  --tracker-source udp --tracker-source-ip WINDOWS_IP \
  --tracker-bind 0.0.0.0 --tracker-port 6210 \
  --tracker-to-wrist-xyz-m 0 0 0 --tracker-to-wrist-rpy-deg 0 0 0 \
  --cmd-addr tcp://127.0.0.1:6100 --img-addr tcp://127.0.0.1:6101 \
  --task-name marvin_left_glove
```

`WINDOWS_IP` 填 Ubuntu 实际收到 UDP 包时的 Windows 来源地址。客户端/服务端都在 Ubuntu，所以它们之间用 127.0.0.1；只有 Tracker 数据跨电脑。手套 SN 与机械手 USB SN 是两个不同设备编号。

保持手套和 Tracker 静止完成初始化。在只读服务端下按 T 会被拒绝，属于预期行为。跟踪失效、报文过期或断流会取消遥操作，不自动恢复。

## 5. 实机运动前的配置与验证

- 测量 Tracker 到手腕的变换，填写客户端的 `--tracker-to-wrist-xyz-m` 和 `--tracker-to-wrist-rpy-deg`。示例零值只是连通检查占位。
- 测量法兰到手部控制点的变换，填写服务端的 `--tool-xyz-m X Y Z` 和 `--tool-rpy-deg R P Y`；不再用 RM75 的法兰厚度。工具变换只在适配器应用一次。
- `--tracking-to-base-rpy-deg R P Y` 将客户端已经转换过的追踪坐标方向映射到机器人工作坐标方向。先离线/只读核对向前、向左、向上的方向。
- 上述工具变换用于主机端运动学。控制器中的工具质量、质心和惯量需按实际 Wuji 手与转接件设置并核对；本次不会用猜测值自动覆盖控制器动力学参数。
- 本路径以当前末端为中心限制 0.15 m 位移、0.05 m/s 平移、20°/s 旋转；驱动限制关节目标变化速率和跟踪误差，并检查 J6/J7 耦合边界。这不是完整碰撞检测，也没有建立实际桌面障碍模型。

完成上述核对后，在服务端命令中增加 **`--enable-motion`**。启动仍保持当前位置，在客户端按 T 才启用臂/手。MARVIN 必须先处于 idle 且无故障；不会自动清错。正常暂停也调用 MARVIN 的软件停止并请求 idle，不是承诺保持姿态的伺服模式。驱动故障会锁定，需排查并重启服务端。

首个测试应由现场人员保持急停可用，以小幅动作验证。这里尚未完成真实 MARVIN、Wuji 左手/手套的联机验收。

## 6. 验证

```bash
cd teleop
python -m unittest tests.test_marvin_adaptation tests.test_network_tracker \
  tests.test_wuji_server_pose_control tests.test_glove_tracker_input \
  tests.test_wuji_driver_diagnostics tests.test_input_recording_contract \
  tests.test_wuji_camera_modes tests.test_wuji_distance_collection -v
```

测试涵盖原生 FK/IK、工具变换、六七关节边界、只读模式、版本不匹配、反馈及指令超时、UDP 重启/乱序/过期消息、左手设备选择及服务端控制回归。原生库不存在时，原生计算测试会跳过；先构建再运行。
