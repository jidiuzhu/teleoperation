# 无 CAD 物体多视角建模与 FoundationPose 使用指南

本文说明 `./run_model_free_foundationpose.sh` 的使用方式。脚本面向没有现成 CAD
模型的物体：使用 Azure Kinect 采集旋转物体的多视角 RGB-D，调用 SAM3 生成
mask，通过 Open3D 配准并重建 mesh，最后把生成的 mesh 注册到本地 C++
FoundationPose 流程。

> “无 CAD”表示不要求用户预先提供 CAD。FoundationPose C++ 的 Register 接口仍然
> 需要 mesh，因此脚本会先自动重建 `model.obj`，而不是把多张图片直接送入 Register。

## 1. 运行前准备

在工作区根目录运行命令：

```bash
cd /media/gc/WorkPlace/code_workspace/rgbd_test
```

需要以下 conda 环境：

```bash
conda env create -f environments/fp-kinect.yml
conda env create -f environments/fp-sam3.yml
conda env create -f environments/fp-reconstruct.yml
```

其中：

- `fp-kinect`：Azure Kinect RGB-D 采集。
- `fp-sam3`：文本或点击式目标分割。
- `fp-reconstruct`：Open3D ICP、点云融合和 Poisson mesh 重建。
- FoundationPose 实时阶段使用 NVIDIA Docker，与前三个进程顺序运行，不同时占用显存。

还需要确认 Azure Kinect、NVIDIA 驱动和 Docker 可用。完整环境安装说明见
[`foundationpose_cpp/LOCAL_DEPLOY.md`](foundationpose_cpp/LOCAL_DEPLOY.md)。

## 2. 推荐的首次建模命令

### 文本提示模式

```bash
./run_model_free_foundationpose.sh \
  --name green_bottle \
  --prompt "green bottle,plastic bottle,green object"
```

`--name` 是对象在 JSON 配置中的唯一名称，建议使用简短的英文小写名称。
`--prompt` 接受逗号分隔的多个候选描述；SAM3 会逐帧选择得分最高的非空 mask。
不要只写过于宽泛的 `object`，否则容易选中桌面上的其他物体。

### 点击选择模式

语言难以描述目标时使用：

```bash
./run_model_free_foundationpose.sh --name special_part --click
```

第一帧会打开 SAM3 交互窗口：

- 左键按当前工具添加正向或负向点。
- 右键始终添加负向点。
- `+`、`-`：切换点类型。
- `Undo` 或 `U`：撤销。
- `Clear` 或 `C`：清空。
- `Confirm` 或 Enter：确认非空 mask。
- `Cancel` 或 Esc：取消。

后续帧根据上一帧目标位置传播正向点，因此物体需要缓慢、连续地旋转，不能突然移出
原位置。点击模式需要图形桌面；`--headless` 只关闭采集和跟踪预览，不会取消 SAM3
点击窗口。

## 3. 正确的采集动作

脚本开始显示采集画面后：

1. 固定 Kinect，相机在整个建模期间不能移动。
2. 让物体尽量保持在图像中央和相近距离。
3. 缓慢旋转物体一整圈。
4. 改变物体俯仰角，再缓慢旋转一整圈，补充顶部和底部视角。
5. 尽量减少手对物体表面的遮挡，不要让其他相似物体进入目标附近。
6. 达到配置帧数后自动结束，也可以按 `Q` 或 Esc 提前结束。

默认每 3 个 Kinect 帧保存 1 帧，共保存 240 帧。30 FPS 相机下约需 24 秒。动作过快
会让相邻视图差异太大；动作过慢或物体完全不转，则无法获得完整背面几何。

## 4. 脚本执行阶段

脚本按顺序执行以下四个阶段：

1. **Capture**：保存对齐的 RGB、米制 depth 和真实相机内参，然后关闭 Kinect。
2. **SAM3**：模型只加载一次，处理全部 RGB，保存 mask 和 overlay，随后彻底退出。
3. **Reconstruction**：过滤 mask 内背景深度，多尺度 ICP 估计相邻物体位姿，融合
   reference cloud，并通过 Poisson 重建 mesh。
4. **Register/Run**：把新对象写入配置；默认继续启动 C++ FoundationPose 实时推理。

SAM3、重建和 FoundationPose 不会同时驻留 GPU。

如果只想生成模型和配置，不立即启动实时相机：

```bash
./run_model_free_foundationpose.sh \
  --name green_bottle \
  --prompt "green bottle,plastic bottle" \
  --build-only
```

之后单独启动实时推理：

```bash
./run_kinect_foundationpose.sh --object green_bottle
```

## 5. 常用参数

| 参数 | 作用 |
|---|---|
| `--name NAME` | 必填，对象配置键和输出目录名。 |
| `--prompt TEXT` | 逗号分隔的 SAM3 文本候选。未指定时使用对象名称。 |
| `--click` | 第一帧使用正负点交互选择目标。 |
| `--segmentation-mode text\|click` | 显式选择分割模式。 |
| `--capture-frames N` | 本次保存的 RGB-D 视图数量，至少为 3。 |
| `--capture-every N` | 每 N 个 Kinect 帧保存一个视图。 |
| `--headless` | 关闭采集和 FoundationPose 跟踪预览。 |
| `--build-only` | 重建并注册对象后停止，不启动实时推理。 |
| `--replace` | 同名对象已存在时更新其 mesh 和 prompts。 |
| `--replay DIR` | 从已有 RGB-D recording 采集，不打开 Kinect。 |
| `--resume-sequence DIR` | 复用已生成的 RGB/depth/mask，仅重跑重建及注册。 |
| `--config FILE` | 使用另一份 pipeline JSON 配置。 |
| `--help` | 显示脚本内置参数帮助。 |

`--replay` 和 `--resume-sequence` 不能同时使用。

例如减少保存帧数进行快速实验：

```bash
./run_model_free_foundationpose.sh \
  --name test_object \
  --click \
  --capture-frames 80 \
  --capture-every 3 \
  --build-only
```

较少帧数适合验证流程，不建议用于最终完整模型。

## 6. 失败后恢复

每次数据保存在：

```text
model_free_objects/<name>/run_YYYYMMDD_HHMMSS/
```

如果 SAM3、ICP 或 mesh 重建中途失败，不需要重新采集。恢复模式会检查连续存在的
mask：mask 尚未完成时从第一个缺失帧继续 SAM3；mask 已完整时不加载 SAM3，直接进入
重建：

```bash
./run_model_free_foundationpose.sh \
  --name my_object \
  --prompt "red apple,red fruit,red object" \
  --resume-sequence model_free_objects/my_object/run_20260818_102926 \
  --build-only
```

若该对象名称已经存在于配置中，增加 `--replace`：

```bash
./run_model_free_foundationpose.sh \
  --name my_object \
  --prompt "red apple,red fruit,red object" \
  --resume-sequence model_free_objects/my_object/run_20260818_102926 \
  --replace \
  --build-only
```

恢复模式要求 sequence 中已有 `meta.json`、`rgb/`、`depth/` 和 `mask/`。

## 7. 输出文件

一个完整 sequence 的结构如下：

```text
run_YYYYMMDD_HHMMSS/
├── meta.json                    # K、深度单位、帧数和采集信息
├── rgb/000000.png               # BGR/RGB 图像序列
├── depth/000000.npy             # 对齐到彩色相机的米制深度
├── mask/000000.png              # SAM3 二值 mask
├── mask/000000_overlay.png      # 分割预览
├── segmentation.json            # 每帧得分、面积和使用的提示方式
└── reconstruction/
    ├── cam_in_ob/000000.txt     # 每帧 4x4 物体到相机位姿
    ├── reference_cloud.ply      # 融合并居中的目标 reference cloud
    ├── model.obj                # 提交给 FoundationPose C++ 的米制 mesh
    └── quality.json             # ICP、点数、mesh 尺寸和质量指标
```

对象注册后，`configs/kinect_foundationpose.json` 中会新增：

```json
"green_bottle": {
  "mesh": "model_free_objects/green_bottle/run_.../reconstruction/model.obj",
  "prompts": ["green bottle", "plastic bottle"],
  "source": "model-free-rgbd"
}
```

脚本同时把该对象设为 `selected_object`。

## 8. 配置参数

长期参数在 `configs/kinect_foundationpose.json` 的 `model_free` 中修改：

```json
"model_free": {
  "output_root": "model_free_objects",
  "capture_frames": 240,
  "capture_every_frames": 3,
  "mask_area_change_ratio": 2.5,
  "max_surface_depth_span_m": 0.2,
  "voxel_m": 0.003,
  "icp_fitness_min": 0.3,
  "icp_rmse_max_m": 0.015,
  "poisson_depth": 8,
  "max_triangles": 100000
}
```

| 配置 | 含义 |
|---|---|
| `capture_frames` | 保存的视图总数。 |
| `capture_every_frames` | 原始相机帧采样间隔。 |
| `mask_area_change_ratio` | 相邻 mask 面积允许的最大倍率变化。 |
| `max_surface_depth_span_m` | mask 内目标可见表面的最大深度跨度，用于剔除后方背景。 |
| `voxel_m` | ICP 和 reference cloud 的体素尺寸，单位米。 |
| `icp_fitness_min` | 相邻帧配准最低重叠率。 |
| `icp_rmse_max_m` | 相邻帧最大配准 RMSE，单位米。 |
| `poisson_depth` | Poisson 重建深度；越高越细致，也越耗时和内存。 |
| `max_triangles` | 输出 mesh 的最大三角形数量。 |

一般先保持默认值。不要只为绕过错误而把 `icp_fitness_min` 大幅降低或把
`icp_rmse_max_m` 大幅提高，这可能生成外观看似完整但坐标错误的 mesh。

## 9. 质量判断

重建成功后检查 `reconstruction/quality.json`：

- `mean_icp_fitness` 越接近 1 越好。
- `max_icp_rmse_m` 应明显低于配置上限。
- `mesh_extent_m` 应与真实物体尺寸接近，单位为米。
- `reference_points`、`mesh_vertices` 和 `mesh_triangles` 不应异常小。

还应查看所有 `mask/*_overlay.png`。稳定的 mask 面积不代表一定选择了正确目标，必须
确认轮廓没有包含手、桌面或相邻物体。

## 10. 常见问题

### `ICP quality failed`

先检查失败帧前后的 overlay。若 mask 正确，通常是物体转动或平移过快；重新采集时
增加动作连续性，或减小 `capture_every_frames`。当前实现会在 point-to-point 和
point-to-plane 之间自动选择稳定结果，不建议直接关闭质量检查。

### `SAM3 mask is empty or too small`

改善提示词、光照和目标可见面积，或改用 `--click`。点击时至少添加一个目标内部正点，
必要时在手和邻近物体上添加负点。

### `SAM3 mask area changed ... x`

目标可能被遮挡、移出画面或分割到了其他物体。查看对应 overlay；修正采集动作或使用
点击模式，不建议盲目提高面积变化阈值。点击模式下自动传播异常时会打开当前帧修正
窗口：清除错误点或继续添加正负点，确认正确 mask 后会从该帧继续。若进程已经退出，
使用 `--resume-sequence` 会从第一个缺失 mask 继续，而不会重做之前的帧。
传播阶段会同时比较纯正点和带外围负点的多个 SAM3 候选，并优先选择与上一帧面积、
位置和重叠关系连续的 mask，而不是仅选择置信度最高的候选；反光表面上的高置信度小块
因此不会自动覆盖完整目标。

### `object 'NAME' already exists`

使用新的 `--name`，或者确认需要覆盖后增加 `--replace`。旧的 sequence 和 mesh 文件
不会删除，配置只会指向新生成的 mesh。

无 CAD 新建流程不依赖当前 `selected_object` 的旧 mesh；即使旧的重建目录已被移动或
删除，仍可使用新的 `--name` 开始采集。常规实时入口仍会严格检查所选 mesh 是否存在。

### `no Open3D reconstruction environment`

```bash
conda env create -f environments/fp-reconstruct.yml
```

### 重建成功但实时 Register 效果差

检查 `mesh_extent_m` 是否与实物一致，确认实时使用的是同一个物体，并在初始化 RGB-D
捕获到 Register 完成之前保持物体不动。必要时重新采集更多顶部、底部和背面视图。
