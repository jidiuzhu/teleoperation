# Kinect 分割点云准备

ACTBH 路线只需 Azure Kinect 对齐 RGB/深度、目标分割点云和已标定的 `T_base_camera`。无需 FoundationPose C++、ONNX/TensorRT、物体网格或 6D 位姿。目录和发布脚本仍保留旧名称，运行时必须加 `--everything`。

## 环境边界

当前版本不支持单一 Python 环境：摇操端的 `dex-retargeting==0.5.0` 依赖 NumPy 2，而本包的 `sam3` 与 `foundationpose-kinect` 包元数据限定 NumPy 1。控制/ACTBH 使用根目录 `requirements.txt`，Kinect 发布使用 `requirements-kinect.txt`。建议控制环境 Python 3.10（本机 Wuji SDK 轮子为 cp310），Kinect 环境 Python 3.11。两个进程在同一主机运行，通过 ZMQ 端口 6200 传输样本，无需共享 Python 包。先按机器 CUDA/CPU 版本分别安装匹配的 PyTorch/torchvision，再安装对应 requirements。

## 设备与模型

1. 按设备许可安装 Azure Kinect Sensor SDK 和 depth engine，参考 `foundationpose-kinect-0.1.0/tools/setup_ubuntu22.sh` 的系统依赖提示，不要未经审查直接运行安装脚本。`requirements-kinect.txt` 包含 `pykinect-azure==0.0.4`，但原生 SDK 和 USB/驱动仍须单独验证。
2. 在 `https://github.com/facebookresearch/sam3` 和 `https://huggingface.co/facebook/sam3` 查看代码、许可及权重访问流程。`requirements-kinect.txt` 会安装包内 SAM3 源码；授权后可执行 `hf auth login` 和 `hf download facebook/sam3 sam3.pt --local-dir foundationpose-kinect-0.1.0/sam3/weights`。检查 `configs/kinect_foundationpose.json` 中 `segmentation.checkpoint`、`bpe` 与实际文件、提示词一致；官方文件名调整时以模型页为准。
3. 需要 Cutie 视频跟踪时，按 `https://github.com/hkchengrex/Cutie` 安装兼容源码；已有验证过的参考版本可用 `python -m pip install --no-deps 'git+https://github.com/hkchengrex/Cutie.git@ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a'`。从 `https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth` 下载权重至 `foundationpose-kinect-0.1.0/weights/`。基础依赖 `hydra-core`、`omegaconf`、`easydict` 已列入 Kinect requirements；若 Cutie 报缺少其他扩展，按其上游安装说明补齐。
4. 用 `python teleop/scripts/calibrate_kinect_rm75_eye_to_hand.py --help` 完成相机到 `base_link` 标定。`teleop/kinect_hand_eye.json` 的初始单位矩阵只是占位，`calibrated` 为 false，不能用于真机。

从项目根目录用原包装脚本准备初始目标。它会先采集 RGB-D、释放相机，再启动 SAM3 点击分割；完成后校验初始文件并退出。`--everything` 强制 `click` 模式，无需另传 `--click`。脚本默认使用名为 `kinect` 的 conda 环境，这里显式指定文档中的 `wuji-kinect`：

```bash
FP_CONDA_ENV=wuji-kinect ./foundationpose-kinect-0.1.0/run_kinect_foundationpose.sh \
  --everything --max-frames 1 --no-pointcloud
```

在弹出的 SAM3 窗口点击目标并确认即可；脚本保存 `foundationpose-kinect-0.1.0/foundationpose_cpp/test_data/live_init/` 下的 `rgb.png`、`depth.png`、`mask.png` 和 `camera.json`。`--no-pointcloud` 只作用于本次初始化，不影响稍后独立运行的点云发布器。换目标或 mask 跟踪漂移后应重新初始化；仅重启发布器不会自动重选目标。该流程无需 FoundationPose mesh 注册，但仍需要这张种子 mask。脚本需要本地图形会话和 `nvidia-smi`。

SAM3 点击窗口中左键选择目标，检查 mask 后按 Enter 确认；单纯点击而未确认不会完成初始化。

发布脚本还读取已标定的 `teleop/kinect_hand_eye.json` 中 `camera_matrix` 作为相机内参；占位文件缺此字段且 `calibrated:false`，因此必须先完成真实手眼/内参标定。然后运行：

```bash
python teleop/scripts/kinect_foundationpose_publisher.py --everything --everything-tracker cutie \
  --init-dir foundationpose-kinect-0.1.0/foundationpose_cpp/test_data/live_init --fps 15
```

未装 Cutie 时可尝试 `--everything-tracker sam3` 低频流式分割；它不是上游完整 SAM3 视频预测器的跨帧实例 ID 流程。需要可视化时加 `--viewer`，Kinect 环境使用 GUI 版 OpenCV。可选 SAM2 mask 精修见 `foundationpose-kinect-0.1.0/tools/install_sam2_tiny.sh` 和发布脚本的 `--sam2`；其权重另行下载。

在 `wuji-actbh` 环境运行采集服务端并加 `--kinect-addr tcp://127.0.0.1:6200 --kinect-hand-eye kinect_hand_eye.json`。`pose_valid=false` 对纯分割模式是正常值；要检查的是样本时效、`segmentation_valid`、点云数量及 15 维距离。两套 requirements 不包括 GPU 驱动、相机原生 SDK 或模型权重。先分别确认相机试采、单帧分割和 HDF5 校验；未接真机的测试不能证明硬件链路可用。
