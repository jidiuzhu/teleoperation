# RM75 + Wuji Hand 遥操作与 ACTBH 模仿学习

输入可选 VR 或 Wuji Glove + Vive Tracker；两者采集相同的 27 维关节状态/命令。训练和部署仅保留 ACTBH：Kinect RGB、深度、分割目标点云、15 个手部关键点到目标点云的最近距离。无 Kinect 录制可检查摇操和回放，但不能直接训练该策略。

## 目录

| 路径 | 用途                                          |
| --- |---------------------------------------------|
| `teleop/scripts/teleop_real_client_wuji.py` | VR/Glove 输入，`--devise vr / glove`           |
| `teleop/scripts/teleop_real_server_wuji.py` | RM75 + Wuji Hand 控制、HDF5 录制                 |
| `teleop/scripts/kinect_foundationpose_publisher.py` | `--everything` 发布 Kinect 分割点云 |
| `teleop/scripts/visualize_rerun.py` | HDF5 图像、点云、关节及距离回放                          |
| `teleop/scripts/convert_h5_to_act.py` | HDF5 校验与 LeRobot 数据集转换                      |
| `teleop/scripts/deploy_actbh_wuji.py` | ACTBH 离线回放、影子推理、受限实机部署                      |
| `third_party/lerobot/` | LeRobot 0.4.5 和本项目 ACTBH/PointNet           |
| `foundationpose-kinect-0.1.0/` | Kinect、SAM3/Cutie 代码；不含原生 SDK/权重            |

## 安装

建议 Ubuntu x86_64。**当前包版本不能单环境安装**：`dex-retargeting==0.5.0` 要求 `numpy>=2`，包内 SAM3 和 
`foundationpose-kinect` 要求 `numpy<2`；OpenCV 4.12 轮子在 Python 3.10/3.11 下也要求 NumPy 2。两套环境用现有 ZMQ 
进程边界通信：`requirements.txt` 用于摇操、转换、ACTBH 训练/部署，`requirements-kinect.txt` 用于 Kinect 分割发布。
本机已安装的 `wuji-sdk`/`wujihandpy` 原生轮子是 `cp310`，因此控制环境示例选 Python 3.10；Kinect 的已安装封装在 Python 3.11 环境。

```bash
conda create -n wuji-actbh python=3.10 -y
conda activate wuji-actbh
conda install --override-channels -c conda-forge pinocchio=3.3.1 casadi=3.6.7 -y
# 按本机 GPU/CPU 选择匹配的 PyTorch、torchvision、torchcodec 安装命令
python -m pip install -r requirements.txt
python -c 'from pinocchio import casadi; import numpy, dex_retargeting, wujihandpy, Robotic_Arm, lerobot; print(numpy.__version__)'

conda create -n wuji-kinect python=3.11 -y
conda activate wuji-kinect
# 按本机 CUDA 版本安装匹配的 PyTorch 和 torchvision
python -m pip install -r requirements-kinect.txt
python -c 'import numpy, sam3, pykinect_azure; print(numpy.__version__)'
```

两套 requirements 只解决 Python 依赖。官方 `wuji-sdk`/`wujihandpy` 须有支持当前 Python 的 wheel；Azure Kinect Sensor 
SDK/depth engine、SteamVR、CUDA/驱动、SAM3/Cutie 权重及视频编解码器仍需按主机安装。
这里只参考其 Glove/Tracker 输入逻辑。详情见 [KINECT.md](KINECT.md)。

`teleop/kinect_hand_eye.json` 默认 `calibrated:false`，禁止将单位矩阵用于真机。用 `teleop/scripts/calibrate_kinect_rm75_eye_to_hand.py --help` 完成相机到 `base_link` 标定。机器人 IP、USB 序列号、Tracker 到腕部外参须按目标机器填写；运动前核对急停、限位、方向。

## 基础摇操

先将服务端源码中的占位 `ROBOT_IP` 改成实际 RM75 地址。在 `teleop/` 目录分别运行：（--flange-thickness按照实际法兰厚度填写）

```bash
python scripts/teleop_real_server_wuji.py --camera-mode none --hand-device YOUR_USB_SERIAL --flange-thickness 0.026
python scripts/teleop_real_client_wuji.py --devise vr --camera-mode none --cmd-addr tcp://SERVER_IP:6100 --img-addr tcp://SERVER_IP:6101 --task-name demo
```

手套 + Vive Tracker 仅替换客户端，wuji动捕手套先完成 Wuji Studio 校准（如果使用wuji studio默认的配置，也可以完成摇操），Vive Tracker需要下载steam，安装steamVR，在steamVR中确认tracker连接正确：

```bash
python scripts/teleop_real_client_wuji.py --devise glove --camera-mode none \
  --glove-sn YOUR_GLOVE_SN --tracker-serial YOUR_TRACKER_SN \
  --tracker-to-wrist-xyz-m 0 0 0 --tracker-to-wrist-rpy-deg 0 0 0 \
  --cmd-addr tcp://SERVER_IP:6100 --img-addr tcp://SERVER_IP:6101 --task-name demo_glove
```

`--devise` 是兼容已有入口的参数名。示例腕部外参零值仅用于检查，实际运动前必须实测。客户端 `T` 启停摇操、`R` 启停录制、`Q` 退出；两端 `--camera-mode` 须一致。同机 `SERVER_IP` 为 `127.0.0.1`。录制到 `teleop/data_records_wuji/`，法兰厚度须实测。VR 需可访问的 Vuer WebXR/HTTPS。无 Kinect 时距离为 NaN，不能用于 ACTBH。

两种手部输入的重定向不同：VR 的 25 点手部关键点使用 `teleop/teleop/wuji.yml` 配置的 DexPilot；该配置沿用旧项目，
默认关闭额外的内部关节约束。Glove 的 21 点骨架交给wuji官方SDK的 `RetargetSession.for_hand(WujiHand, Right)` 输出 20 维关节目标，
Vive Tracker 提供腕部位姿；glove输入不经过 VR 的 DexPilot。两种输入最终进入同一服务端控制和 HDF5 数据契约。

## Kinect 采集

按 [KINECT.md](KINECT.md) 准备 Azure Kinect 和 SAM3/Cutie 权重。**发布前仍须点击选择初始目标 mask**。旧项目的包装脚本会依次采一帧 RGB-D、运行 SAM3 点击分割并校验 mask；`--everything` 初始化完成后直接退出，不启动 FoundationPose。发布进程用 `wuji-kinect` 环境，采集服务端用 `wuji-actbh` 环境；两进程应在同一主机以校验单调时钟。初始化示例（脚本通过 `FP_CONDA_ENV` 选择环境）：

```bash
FP_CONDA_ENV=wuji-kinect ./foundationpose-kinect-0.1.0/run_kinect_foundationpose.sh \
  --everything --max-frames 1 --no-pointcloud
```

确保 `teleop/kinect_hand_eye.json` 已完成标定且包含 `camera_matrix`，然后启动发布：

```bash
python teleop/scripts/kinect_foundationpose_publisher.py --everything --everything-tracker cutie \
  --init-dir foundationpose-kinect-0.1.0/foundationpose_cpp/test_data/live_init --fps 15
```

可把 `cutie` 改为 `sam3`，但是sam3实时推理很慢，cutie实时速率高。在 `teleop/` 运行上述服务端时增加 `--kinect-addr tcp://127.0.0.1:6200 --kinect-hand-eye kinect_hand_eye.json`；客户端仍用所选 `--devise`。`--everything` 中 `pose_valid=false` 正常，数据有效性由 Kinect 时效、分割有效性、非空点云和有限距离判断。失效帧可能被录下用于诊断；转换器拒绝整条含失效帧的示教，需重录或另行清洗，不可用假位姿/零点云填充。

Kinect 环境使用 GUI 版 OpenCV，可按需增加 `--viewer`；控制/训练环境使用 headless 版，避免同一环境混装两种 OpenCV 轮子。

## 回放、转换与训练

```bash
python teleop/scripts/visualize_rerun.py teleop/data_records_wuji/YOUR_EPISODE.h5 --frame-step 2
python teleop/scripts/convert_h5_to_act.py --source teleop/data_records_wuji/YOUR_EPISODE.h5 --repo-id local/wuji_act --validate-only
python teleop/scripts/convert_h5_to_act.py --source teleop/data_records_wuji/YOUR_EPISODE.h5 --repo-id local/wuji_act --target-dir teleop/lerobot_datasets/wuji_act
```

输入：`observation.state` `[27]`、`observation.environment_state` `[15]`（米）、RGB/深度各 `[3,240,320]`、
基座系点云 `[1024,3]`；动作 `[27]`。距离由采集端按分割点云计算，转换器校验有限性，不读取 FoundationPose。旧 24 维环境状态检查点不兼容，
需重新训练。输入设备仅标记数据来源，不改变特征。

```bash
python third_party/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/wuji_act --dataset.root=teleop/lerobot_datasets/wuji_act \
  --policy.type=actbh --policy.device=cuda --policy.repo_id=local/wuji_actbh \
  --policy.input_features='{"observation.state":{"type":"STATE","shape":[27]},"observation.environment_state":{"type":"ENV","shape":[15]},"observation.images.KINECT_rgb":{"type":"VISUAL","shape":[3,240,320]},"observation.images.KINECT_depth":{"type":"VISUAL","shape":[3,240,320]},"observation.point_cloud_global":{"type":"STATE","shape":[1024,3]}}' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[27]}}' \
  --policy.pretrained_backbone_weights=null --steps=20000 --batch_size=8
```

按 GPU 内存及样本量调整训练参数。先离线回放部署，不连接硬件：

```bash
python teleop/scripts/deploy_actbh_wuji.py --checkpoint outputs/train/latest/pretrained_model \
  --replay-h5 teleop/data_records_wuji/YOUR_EPISODE.h5 --replay-frames 20
```

在线部署需已标定 Kinect 分割流、机械臂及手反馈。默认仅影子推理；实机运动需显式传 `--enable-motion --motion-confirmation=I_UNDERSTAND_THIS_MOVES_THE_ROBOT`。先核对回放误差、时效、速度和工作空间，软件限幅不能替代急停。完整参数见 `python teleop/scripts/deploy_actbh_wuji.py --help`。

## 验证

```bash
cd teleop
python -m unittest tests.test_convert_h5_to_act tests.test_actbh_deploy -v
```
