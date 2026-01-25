# VITRA 数据自动标注流程

本文档提供了 VITRA 数据集自动化标注流程的详尽指南。该流程旨在将原始视频转化为完全标注的、包含3D重建信息和动作分段的数据集，适用于机器人学习和动作识别模型的训练。

## 1. 流程总览 (Pipeline Overview)

该流程通过一系列连续的处理阶段，将原始视频转换为丰富的语义和几何数据：

1.  **视频处理 (Video Processing)**：帧提取、尺寸调整和去畸变。
2.  **手部重建 (Hand Reconstruction)**：相机空间下的3D手部位姿估计 (使用 HaWoR/MANO)。
3.  **相机位姿估计 (Camera Pose Estimation)**：使用 SLAM 技术跟踪相机运动轨迹。
4.  **世界坐标系转换 (World-Space Transformation)**：将相机空间的手部数据转换到世界坐标系。
5.  **动作分割 (Action Segmentation)**：基于运动启发式算法（速度极小值法），将连续视频分割为若干个原子动作片段。
6.  **语言标注 (Language Annotation)**：利用大型语言模型 (LLM) 为每个动作片段生成自然语言描述。
7.  **可视化 (Visualization)**：在视频上渲染3D覆盖层和文本标注，用于人工验证。
8.  **格式化 (Formatting)**：将元数据保存为 VITRA 标准的 `.npy` 格式。

### 为什么需要世界坐标系 (World-Space)?

第一人称视频中相机会移动，获取世界坐标系的手部数据至关重要：
- 可以将动作重新投影到任意帧的相机视角
- 模拟机器人操作时的"静止相机"视角
- 将晃动的人类视频变成稳定的机器人训练数据

## 2. 目录结构 (Directory Structure)

```
data/pipeline/
├── config.py             # 核心配置管理 (使用 dataclasses)
├── builder.py            # 主调度器 (DatasetBuilder 类)
├── stages/               # 各个独立的处理阶段
│   ├── base.py                 # 阶段的抽象基类
│   ├── video_processor.py      # 阶段 1: 视频 I/O
│   ├── hand_reconstruction.py  # 阶段 2: 3D 重建 (Camera-Space)
│   ├── camera_pose.py          # 阶段 3: 相机位姿估计 (SLAM)
│   ├── action_segmentation.py  # 阶段 5: 时序分割
│   ├── language_annotation.py  # 阶段 6: 文本生成
│   └── visualization.py        # 阶段 7: 渲染与输出
└── ...
```

## 3. 配置说明 (Configuration)

该流程可以通过 `data/pipeline/config.py` 进行高度配置。您可以在运行时提供自定义的 YAML 配置文件，或直接在代码中修改默认值。推荐使用 yaml 配置文件。

### 关键配置项：

-   **`VideoConfig`** (视频配置):
    -   `supported_formats`: 支持的文件扩展名列表 (例如 `['.mp4', '.avi']`)。
    -   `target_fps`: 重采样目标帧率 (默认: None，保持原始帧率)。
    -   `resize_short_side`: 调整输入帧的短边尺寸 (例如 480)。

-   **`HandReconConfig`** (手部重建配置 - 阶段 2):
    -   `detection_threshold`: 手部检测置信度阈值 (默认 0.2)。
    -   `detector_path`: 手部检测器模型路径。
    -   `mano_path`: MANO 手部模型权重路径。
    -   `batch_size`: GPU 推理的批处理大小。

-   **`SegmentationConfig`** (分割配置 - 阶段 3):
    -   `smoothing_sigma`: 轨迹高斯平滑的强度。
    -   `minima_window_size`: 检测分割点（速度极小值）的时间窗口大小（秒）。
    -   `min_segment_duration`: 动作片段的最小持续时间（秒）。
    -   `min_motion_threshold`: 动作片段的最小移动距离（米），用于过滤静止手/无效片段（默认 0.01）。

-   **`AnnotationConfig`** (标注配置 - 阶段 4):
    -   `llm_provider`: `openai`, `azure` 或 `local`。
    -   `llm_model`: 模型名称 (例如 `gpt-4o`)。
    -   `action_prompt_template`: 生成动作描述的 Prompt 模板。
    -   `api_key`: (可选) LLM 提供商的 API Key。

-   **`OutputConfig`** (输出配置):
    -   `save_visualization`: 是否渲染输出视频。
    -   `compress_output`: 是否压缩 numpy 数组。

## 4. 使用指南 (Usage)

要处理目录中的视频，请使用 `scripts/run_annotation_pipeline.py` 脚本。

### 基础命令：

支持处理**单个视频文件**或**包含多个视频的文件夹**。

**处理整个文件夹：**
```bash
python scripts/run_annotation_pipeline.py \
    --input_dir data/examples/videos \
    --output_dir data/my_dataset
```

**处理单个视频：**
```bash
python scripts/run_annotation_pipeline.py \
    --input_dir data/examples/videos/demo.mp4 \
    --output_dir data/my_dataset
```

### 使用自定义配置和 GPU：

```bash
python scripts/run_annotation_pipeline.py \
    --input_dir data/raw_videos \
    --output_dir data/processed_vitra \
    --config configs/my_pipeline_config.yaml \
    --device cuda \
    --verbose
```

## 5. 算法细节 (Algorithm Details)

### 动作分割 (基于速度极小值)
流程实现了 VITRA 论文中描述的分割启发式算法：
1.  **提取轨迹**：从重建阶段提取3D手腕位置。
2.  **平滑处理**：使用高斯滤波器 (`smoothing_sigma`) 对轨迹进行平滑，消除抖动。
3.  **计算速度**：计算每一帧的3D速度矢量的模长。
4.  **极小值检测**：在时间窗口 (`minima_window_size`) 内寻找速度曲线的局部极小值。
5.  **切分**：将这些极小值点作为"原子动作"的起始/结束边界。

### 语言标注 (Language Annotation)
1.  **运动摘要**：为每个片段计算手部运动的统计摘要（位移、方向、起始/结束位置）。
2.  **Prompt 填充**：将摘要填入 `action_prompt_template` 并发送给 LLM。
3.  **生成结果**：LLM 返回简洁的动作描述（例如 "拿起桌上的苹果"）。

## 6. 输出结构 (Output Structure)

在您指定的 `output_dir` 目录下：

```
output_dir/
├── episodic_annotations/
│   ├── {dataset}_{video}_ep_{id}.npy  # 符合 VITRA 格式的标准数据文件
│   └── ...
└── vis/                                # 可视化结果
    └── {video_name}/                   # 每个源视频单独一个文件夹
        ├── full.mp4                    # 完整的带标注视频
        ├── ep_000000.mp4               # 单独的动作片段 0
        ├── ep_000001.mp4               # 单独的动作片段 1
        └── ...
```

### `.npy` 格式说明

每个 `.npy` 文件对应一个原子动作片段，包含一个字典。完整的字段规范请参考 [`data/data.md`](../data.md) 第 4 节。

#### 顶层字段 (Top-level Fields)

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `video_clip_id_segment` | `np.ndarray` | `(T,)` int64 | 视频片段 ID (已弃用，保留兼容性) |
| `extrinsics` | `np.ndarray` | `(T, 4, 4)` float64 | World2Cam 外参矩阵 |
| `intrinsics` | `np.ndarray` | `(3, 3)` float64 | 相机内参矩阵 |
| `video_decode_frame` | `np.ndarray` | `(T,)` int64 | 原始视频中的帧索引 |
| `video_name` | `str` | - | 源视频名称 |
| `avg_speed` | `float` | 标量 | 每帧平均手腕移动距离（米） |
| `total_rotvec_degree` | `float` | 标量 | 片段内相机总旋转角度（度） |
| `total_transl_dist` | `float` | 标量 | 片段内相机总平移距离（米） |
| `anno_type` | `str` | - | 标注类型 (`'left'` 或 `'right'`) |
| `text` | `dict` | - | 文本标注 `{'left': [(描述, 范围)], 'right': [...]}` |
| `text_rephrase` | `dict` | - | GPT-4 改写的文本标注 |
| `left` | `dict` | - | 左手 3D 位姿数据 |
| `right` | `dict` | - | 右手 3D 位姿数据 |

#### 手部数据字段 (`left`/`right` 子字典)

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `beta` | `np.ndarray` | `(10,)` float64 | MANO 手型参数 |
| `global_orient_camspace` | `np.ndarray` | `(T, 3, 3)` float64 | 相机空间中的手腕旋转矩阵 |
| `global_orient_worldspace` | `np.ndarray` | `(T, 3, 3)` float64 | 世界空间中的手腕旋转矩阵 |
| `hand_pose` | `np.ndarray` | `(T, 15, 3, 3)` float64 | 15 个关节的局部旋转矩阵 |
| `transl_camspace` | `np.ndarray` | `(T, 3)` float64 | 相机空间中的手腕平移 (已弃用) |
| `transl_worldspace` | `np.ndarray` | `(T, 3)` float64 | 世界空间中的手腕平移 |
| `kept_frames` | `np.ndarray` | `(T,)` int64 | 有效帧掩码 (0/1) |
| `joints_camspace` | `np.ndarray` | `(T, 21, 3)` float32 | 相机空间中的 21 个关节 3D 位置 |
| `joints_worldspace` | `np.ndarray` | `(T, 21, 3)` float64 | 世界空间中的 21 个关节 3D 位置 |
| `wrist` | `np.ndarray` | `(T, 1, 3)` float32 | 手腕位置 (已弃用) |
| `max_translation_movement` | `float` 或 `None` | 标量 | 最大帧间手腕位移 |
| `max_wrist_rotation_movement` | `float` 或 `None` | 标量 | 最大帧间手腕旋转角度 |
| `max_finger_joint_angle_movement` | `float` 或 `None` | 标量 | 最大帧间手指关节角度变化 |

## 7. 数据格式验证 (Format Validation)

我们提供了一个验证工具来检查生成的 `.npy` 文件是否符合 VITRA 标准格式。

### 验证单个文件：
```bash
python my_demo/utils/validate_npy_format.py --input path/to/file.npy
```

### 验证整个目录：
```bash
python my_demo/utils/validate_npy_format.py --input data/processed_dataset/episodic_annotations
```

### 查看文件结构：
```bash
python my_demo/utils/validate_npy_format.py --input path/to/file.npy --structure
```

### 详细输出模式：
```bash
python my_demo/utils/validate_npy_format.py --input path/to/file.npy --verbose
```

验证工具会检查：
- 所有必需字段是否存在
- 数据类型是否正确 (dtype)
- 数组形状是否符合规范
- 并生成详细的验证报告

## 8. 扩展流程 (Extending the Pipeline)

如需添加新功能：
1.  创建一个继承自 `stages.base.TimedStage` 的新类。
2.  实现 `_do_initialize()` 和 `_do_process()` 方法。
3.  将该阶段添加到 `builder.py` 的 `__init__` 和 `process_single_video` 序列中。
4.  在 `config.py` 中添加相应的配置类。
