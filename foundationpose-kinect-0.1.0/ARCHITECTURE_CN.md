# FoundationPose Kinect 模块架构

## 1. 目标与边界

本项目的本地主路径是 Ubuntu 22.04 + Azure Kinect DK + SAM3 + FoundationPose C++。
模块化后分为两种调用方式：

- **库模式**：其他 Python 模块提供 RGB-D 和初始化 mask，同步取得 pose 与目标点云。
- **托管流程模式**：复用现有相机、SAM3、Docker 和可视化一键流程，外部模块负责启停并读取产物。

FoundationPose C++ 的 TCP 协议保持不变。模块层没有引入网络代理、数据库、ROS 或新的
序列化格式。

## 2. 分层架构

```text
调用方业务模块
├── FoundationPoseSession       同进程、逐帧同步 API
├── FoundationPoseService       C++ Docker 服务生命周期
└── PipelineProcess             完整一键流程生命周期
        │
        ├── RGBDFrame / PoseResult / PointCloud
        ├── ProjectConfig / RunArtifacts
        │
        ├── Azure Kinect capture (fp-kinect)
        ├── SAM3 initialization (fp-sam3，完成后退出)
        ├── Open3D model-free reconstruction (fp-reconstruct)
        │
        └── FoundationPose C++ TCP service (NVIDIA Docker)
                ├── Register: RGB + aligned depth + mask + K
                └── Track:    RGB + aligned depth + K
```

## 3. 目录职责

| 路径 | 职责 |
|---|---|
| `src/foundationpose_kinect/` | 可安装的稳定公共 Python API。 |
| `foundationpose_cpp/` | TensorRT/CUDA C++ 推理服务、Dockerfile 和 ONNX。 |
| `server/sam3_*.py` | 单帧和批量 SAM3 分割，运行于独立环境。 |
| `kinect_foundationpose_live.py` | 现有完整实时采集、预览和回放客户端。 |
| `run_kinect_foundationpose.sh` | CAD/已重建 mesh 对象的一键实时流程。 |
| `run_model_free_foundationpose.sh` | 无 CAD 多视角采集、分割、重建和注册。 |
| `configs/` | 对象、运行时、点云与无 CAD 参数。 |
| `environments/` | 三个可复现 conda 环境。 |
| `tools/build_release_bundle.sh` | 生成可分享发布包。 |

旧的远端 Python 服务代码仍保留在 `server/`，但不属于 Ubuntu 本地 C++ 主运行依赖。

## 4. 公共数据契约

### RGBDFrame

- `color_bgr`: `uint8[H,W,3]`
- `depth_mm`: `uint16[H,W]`，已对齐到彩色相机
- `intrinsics`: `float[3,3]`，真实彩色相机内参
- `frame_id`: 可选；省略时 session 自动递增

### PoseResult

- `pose`: `float32[4,4]`，物体坐标到相机坐标的刚体变换
- `translation_m`: 米制平移快捷属性
- `inference_ms`: C++ Register/Track 耗时
- `observed`: 当前 RGB-D 裁剪出的目标点云
- `mesh`: 当前 pose 下的 mesh 表面点云
- `reference`: 初始化若干观测融合得到的参考点云
- `clouds_updated`: 本帧是否实际更新点云

三类点云均使用相机坐标系、米制 XYZ 和 RGB uint8 颜色。

## 5. 性能设计

模块化不改变 C++ 推理协议，也不修改现有实时脚本热路径：

- pose-only 模式不加载 mesh，不执行深度反投影，也不融合 reference cloud。
- NumPy 输入满足 dtype 和连续布局时，协议层不额外复制。
- pose 使用固定 88 字节二进制响应，不经过 JSON 或文件。
- 点云仅在明确传入 `PointCloudConfig` 时计算。
- `pointcloud_every=N` 可让每 N 帧更新一次点云，其余帧只返回 pose。
- `track(..., with_pointcloud=False)` 可临时跳过单帧点云。
- mesh 只在 session 创建时解析和采样一次。

需要跨语言或跨进程高频发布时，应在现有二进制协议上新增共享内存/ROS2 适配器，而不应
让核心 session 每帧写 PLY。当前 `RunArtifacts` 适合低频轮询和流程集成，不是高频 IPC。

## 6. 生命周期约束

RTX 4060 Laptop 8 GB 等显存较小的设备必须保持：

```text
Kinect capture -> SAM3 -> SAM3 exit -> FoundationPose C++
```

无 CAD 路径为：

```text
capture sequence -> SAM3 batch -> SAM3 exit -> Open3D -> C++ Register/Track
```

`FoundationPoseService` 只管理 C++ 容器，不会隐式启动 SAM3。调用方需要提供已确认的
初始化 mask，这使它可以接入 SAM3、其他分割模型、人工标注或上游检测系统。

托管实时脚本另提供默认关闭的 `--hybrid` 模式。该模式在独立子进程中常驻 SAM3 image
model，以有界的最新帧队列低频检查目标 mask；FoundationPose 仍每帧 Track，只在 mask
中心和 pose 投影中心偏差明显时按需重新 Register。它不是 SAM3 官方视频 session，因而
不会积累完整视频特征。该模式要求同时容纳 PyTorch 和 TensorRT，建议 12 GB 或更高显存。

## 7. 兼容性

- TCP 协议版本仍为 1，现有 C++ 服务无需因 Python 包而改变。
- 现有 Bash 命令保持可用。
- TensorRT engine 不进入发布包，目标机器根据本机 GPU、TensorRT 和 ONNX 重建。
- 对象 mesh 和深度单位统一为米。
