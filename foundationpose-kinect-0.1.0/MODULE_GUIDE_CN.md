# FoundationPose Kinect 模块调用指南

## 1. 安装公共 Python 包

在工作区根目录执行：

```bash
conda run -n fp-kinect python -m pip install --no-deps -e .
```

验证：

```bash
conda run -n fp-kinect fp-kinect inspect-config \
  --root . --object apple_10cm
```

## 2. 库模式：逐帧取得 pose 和点云

该模式适合机器人、抓取、测量或其他视觉模块。调用方负责提供初始化 mask。

```python
import numpy as np

from foundationpose_kinect import (
    FoundationPoseService,
    FoundationPoseSession,
    PointCloudConfig,
    ProjectConfig,
    RGBDFrame,
)

ROOT = "/path/to/foundationpose-kinect"
config = ProjectConfig(f"{ROOT}/configs/kinect_foundationpose.json", ROOT)
obj = config.object("apple_10cm")

# 初始 RGB-D 和 mask 可来自 Azure Kinect、录制回放或其他模块。
initial = RGBDFrame(color_bgr, depth_mm, K)
mask = target_mask_uint8

with FoundationPoseService(ROOT, object_name=obj.name) as service:
    # 不需要点云时将 pointcloud_config 改为 None，即 pose-only 快速路径。
    with FoundationPoseSession(
        port=service.port,
        mesh_path=obj.mesh_path,
        pointcloud_config=config.pointcloud_config(),
        pointcloud_every=3,
    ) as session:
        result = session.register(initial, mask)
        consume_pose(result.pose, result.inference_ms)

        for color_bgr, depth_mm, K in frame_source:
            result = session.track(RGBDFrame(color_bgr, depth_mm, K))
            consume_pose(result.pose, result.inference_ms)
            if result.clouds_updated:
                consume_observed(result.observed.points, result.observed.colors_rgb)
                consume_mesh(result.mesh.points, result.mesh.colors_rgb)
                consume_reference(result.reference.points, result.reference.colors_rgb)
```

### 最低开销 pose-only

```python
with FoundationPoseSession(port=service.port) as session:
    result = session.register(initial, mask)
    result = session.track(frame)
    pose = result.pose
```

此模式不要求 `mesh_path`，因为 mesh 已由 C++ 服务加载，Python 端不做点云计算。

### 单帧控制点云

```python
pose_only = session.track(frame, with_pointcloud=False)
pose_and_clouds = session.track(next_frame, with_pointcloud=True)
```

`FoundationPoseSession` 是同步且线程安全的，同一个 session 内 Register/Track 严格串行。
业务侧可把结果送入自己的无锁队列，但不要并发调用同一 C++ 连接。

### 性能边界与实测

- `pointcloud_config=None` 时不加载 Python mesh，不反投影深度，也不融合 reference cloud。
- 模块直接复用现有二进制 TCP 协议；输入布局连续时不会先创建另一份 NumPy 数组。
- 点云需要时建议从 `pointcloud_every=2` 或 `3` 开始，按业务刷新率逐步调整。
- 不要在逐帧回调中使用 `RunArtifacts` 读取 PLY；它只适合低频流程集成。

2026-08-18 在 RTX 4060 Laptop 上用现有录制回放 30 帧，关闭点云后的原启动路径为
33.63 FPS，Track 平均 4.18 ms，峰值显存 5682 MiB，最终 pose 非单位阵。该结果处于改造前
约 32–35 FPS 的范围内。指标保存在
`foundationpose_cpp/test_output/live/run_20260818_115756/metrics.json`。

## 3. 托管完整流程

若其他模块只需要启停现有 Kinect + SAM3 + FoundationPose 流程：

```python
import time
from foundationpose_kinect import PipelineProcess

pipeline = PipelineProcess("/path/to/foundationpose-kinect")
pipeline.start("--object", "apple_10cm", "--headless")

try:
    while pipeline.running:
        artifacts = pipeline.artifacts()
        if artifacts and artifacts.pose_path.is_file():
            pose = artifacts.latest_pose()
            observed = artifacts.latest_cloud("observed")
            if observed is not None:
                consume(observed.points)
        time.sleep(0.1)
finally:
    pipeline.stop()
```

该模式的 pose 文件每帧原子更新；PLY 按配置中的 `save_every_frames` 更新。需要每帧低延迟
点云时应使用库模式，不要轮询 PLY。

## 4. 命令行接口

```bash
# 检查对象配置
fp-kinect inspect-config --root . --object apple_10cm

# 使用模块入口启动现有一键流程，`--` 后为原脚本参数
fp-kinect run --root . -- --object apple_10cm --headless --max-frames 100

# 查看某次运行的最新 pose 和 metrics
fp-kinect artifacts --run foundationpose_cpp/test_output/live/run_YYYYMMDD_HHMMSS
```

## 5. 异常与清理

- `FoundationPoseService` 和 `FoundationPoseSession` 均支持 `with`，退出时清理容器/连接。
- Register 前必须提供非空且尺寸匹配的 mask。
- Track 的图像尺寸不能在同一 session 中变化。
- 相机内参、depth 和 mesh 必须使用一致的米制几何。
- C++ 服务中断、协议状态非零或超时会抛出 Python 异常，不返回伪造 pose。

完整架构见 [ARCHITECTURE_CN.md](ARCHITECTURE_CN.md)，跨机器安装见
[DEPLOYMENT_CN.md](DEPLOYMENT_CN.md)。
