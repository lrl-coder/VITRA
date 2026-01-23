# VITRA 数据自动标注流程

本文档提供了 VITRA 数据集自动化标注流程的详尽指南。该流程旨在将原始视频转化为完全标注的、包含3D重建信息和动作分段的数据集，适用于机器人学习和动作识别模型的训练。

## 1. 流程总览 (Pipeline Overview)

该流程通过一系列连续的处理阶段，将原始视频转换为丰富的语义和几何数据：

1.  **视频处理 (Video Processing)**：帧提取、尺寸调整和去畸变。
2.  **手部重建 (Hand Reconstruction)**：3D手部和相机位姿估计 (使用 HaMeR/MANO 等技术)。
3.  **动作分割 (Action Segmentation)**：基于运动启发式算法（速度极小值法），将连续视频分割为若干个原子动作片段。
4.  **语言标注 (Language Annotation)**：利用大型语言模型 (LLM) 为每个动作片段生成自然语言描述。
5.  **可视化 (Visualization)**：在视频上渲染3D覆盖层和文本标注，用于人工验证。
6.  **格式化 (Formatting)**：将元数据保存为 VITRA 标准的 `.npy` 格式。

## 2. 目录结构 (Directory Structure)

```
data/pipeline/
├── config.py             # 核心配置管理 (使用 dataclasses)
├── builder.py            # 主调度器 (DatasetBuilder 类)
├── stages/               # 各个独立的处理阶段
│   ├── base.py                 # 阶段的抽象基类
│   ├── video_processor.py      # 阶段 1: 视频 I/O
│   ├── hand_reconstruction.py  # 阶段 2: 3D 重建
│   ├── action_segmentation.py  # 阶段 3: 时序分割
│   ├── language_annotation.py  # 阶段 4: 文本生成
│   └── visualization.py        # 阶段 5: 渲染与输出
└── ...
```

## 3. 配置说明 (Configuration)

该流程可以通过 `data/pipeline/config.py` 进行高度配置。您可以在运行时提供自定义的 YAML 配置文件，或直接在代码中修改默认值。

### 关键配置项：

-   **`VideoConfig`** (视频配置):
    -   `supported_formats`: 支持的文件扩展名列表 (例如 `['.mp4', '.avi']`)。
    -   `target_fps`: 重采样目标帧率 (默认: None，保持原始帧率)。
    -   `resize_short_side`: 调整输入帧的短边尺寸 (例如 480)。

-   **`HandReconConfig`** (手部重建配置 - 阶段 2):
    -   `detector_path`: 手部检测器模型路径。
    -   `mano_path`: MANO 手部模型权重路径。
    -   `batch_size`: GPU 推理的批处理大小。

-   **`SegmentationConfig`** (分割配置 - 阶段 3):
    -   `smoothing_sigma`: 轨迹高斯平滑的强度。
    -   `minima_window_size`: 检测分割点（速度极小值）的时间窗口大小（秒）。
    -   `min_segment_duration`: 动作片段的最小持续时间（秒）。

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

```bash
python scripts/run_annotation_pipeline.py \
    --input_dir data/examples/videos \
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
每个 `.npy` 文件对应一个原子动作片段，包含一个字典：
-   `video_name`: 源视频名称。
-   `video_decode_frame`: 该片段对应的帧索引列表。
-   `text`: `{'left': [(描述, 范围)], 'right': [...]}` 文本标注。
-   `left`/`right`: 包含该手部 3D 位姿数据 (`hand_pose`, `transl`, `beta`) 的字典。
-   `extrinsics`/`intrinsics`: 相机参数。

## 7. 扩展流程 (Extending the Pipeline)

如需添加新功能：
1.  创建一个继承自 `stages.base.TimedStage` 的新类。
2.  实现 `_do_initialize()` 和 `_do_process()` 方法。
3.  将该阶段添加到 `builder.py` 的 `__init__` 和 `process_single_video` 序列中。
4.  在 `config.py` 中添加相应的配置类。
