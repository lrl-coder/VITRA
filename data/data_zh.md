# 人类手部 V-L-A 数据准备说明

本文件夹提供了本项目中用于人类手部 V-L-A（视觉-语言-动作）数据的基本文档和脚本。
**请注意，我们要提供的元数据（Metadata）未来可能会持续更新。基于人工检查，当前版本大约达到了 90% 的标注准确率，我们计划在未来的更新中进一步提高元数据质量。**

本文件夹的内容如下：

## 📑 目录
- [1. 前置要求](#1-前置要求)
- [2. 数据下载](#2-数据下载)
- [3. 视频预处理](#3-视频预处理)
- [4. 元数据结构](#4-元数据结构)
- [5. 数据可视化](#5-数据可视化)

---
## 1. 前置要求
我们的数据预处理和可视化依赖于一些需要预先准备的库。如果您已经完成了 [`readme.md`](../readme.md) 中 **1.2 可视化要求** 的安装步骤，则可以跳过本节。

### Python 库
可视化需要 [PyTorch3D](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file)。您可以按照官方指南安装，或者直接运行以下命令：
```bash
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable#egg=pytorch3d  
```
视频处理还需要 [FFmpeg](https://github.com/FFmpeg/FFmpeg)：
```bash
sudo apt install ffmpeg
pip install ffmpeg-python
```

其他 Python 依赖项可以使用以下命令安装：
```bash
pip install projectaria_tools smplx
pip install --no-build-isolation git+https://github.com/mattloper/chumpy#egg=chumpy
```
### MANO 手部模型

我们重建的手部标签是基于 MANO 手部模型的。**我们只需要右手模型。** 模型参数可以从 [官方网站](https://mano.is.tue.mpg.de/index.html) 下载，并按以下结构组织（[mano_mean_params.npz](../weights/mano/mano_mean_params.npz) 已包含在我们的代码库中）：
```
weights/
└── mano/
    ├── MANO_RIGHT.pkl
    └── mano_mean_params.npz
```

---

## 2. 数据下载

### 元信息 (Meta Information)

我们提供了构建的人类 V-L-A 片段 (episodes) 的元数据，可以从 [此链接](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) 下载。每个元数据条目包含相应 V-L-A 片段的分割信息、语言描述以及重建的相机参数和 3D 手部信息。元数据的详细结构可以在 [元数据结构](#4-元数据结构) 中找到。所有元数据的总大小约为 100 GB。

解压后，下载的元数据将具有以下结构：
```
Metadata/
├── {dataset_name1}/
│   ├── episode_frame_index.npz
│   └── episodic_annotations/
│       ├── {dataset_name1}_{video_name1}_ep_{000000}.npy
│       ├── {dataset_name1}_{video_name1}_ep_{000001}.npy
│       ├── {dataset_name1}_{video_name1}_ep_{000002}.npy
│       ├── {dataset_name1}_{video_name2}_ep_{000000}.npy
│       ├── {dataset_name1}_{video_name2}_ep_{000001}.npy
│       └── ...
├── {dataset_name2}/
│   └── ...
```
这里，`{dataset_name}` 表示片段所属的数据集，`{video_name}` 对应原始视频的名称，`ep_{000000}` 是片段的索引。

### 视频

我们的项目目前使用从四个来源收集的视频：[Ego4D](https://ego4d-data.org/#)、[Epic-Kitchen](https://epic-kitchens.github.io/2025)、[EgoExo4D](https://ego-exo4d-data.org/#intro) 和 [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)。由于许可限制，我们不能直接提供我们处理后的视频数据。如需访问数据，请申请并从官方数据集网站下载原始视频。请注意，本项目只需要 *原始视频 (raw video)* 文件。

每个数据集下载的原始数据的结构如下：
- **Ego4D**:  
```
Ego4D_root/
└── v2/
    └── full_scale/
        ├── {video_name1}.mp4
        ├── {video_name2}.mp4
        ├── {video_name3}.mp4
        └── ...
```
- **Epic-Kitchen**:  
```
Epic-Kitchen_root/
├── P01/
│   └── videos/
│       ├── {video_name1}.MP4
│       ├── {video_name2}.MP4
│       └── ...
├── P02/
│   └── videos/
│       ├── {video_name3}.MP4
│       ├── {video_name4}.MP4
│       └── ...
└── ...
```
- **EgoExo4D**:  
```
EgoExo4D_root/
└── takes/
    ├── {video_name1}/
    │   └── frame_aligned_videos/
    │       ├── {cam_name1}.mp4
    │       ├── {cam_name2}.mp4
    │       └── ...
    ├── {video_name2}/
    │   └── frame_aligned_videos/
    │       ├── {cam_name1}.mp4
    │       ├── {cam_name2}.mp4
    │       └── ...
    └── ...
```
- **Somethingsomething-v2**:  
```
Somethingsomething-v2_root/
├── {video_name1}.webm
├── {video_name2}.webm
├── {video_name3}.webm
└── ...
```
---

## 3. 视频预处理

Ego4D 和 EgoExo4D 中的大部分原始视频都存在鱼眼畸变。为了标准化处理，我们要校正鱼眼畸变并将视频转换为针孔相机模型。我们的元数据是基于最终去畸变后的视频的。为了能够复现我们的数据，我们提供了对原始视频执行此去畸变操作的脚本。

### 相机内参 (Camera Intrinsics)

我们提供了 Ego4D 原始视频的估计内参（如我们论文中所述，使用 [DroidCalib](https://github.com/boschresearch/DroidCalib) 计算）和 EgoExo4D 的真实 Project Aria 内参（来自 [官方仓库](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation)）。这些文件可以通过 [此链接](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M/tree/main/intrinsics) 下载，并按以下方式组织：
```
camera_intrinsics_root/
├── ego4d/
│   ├── {video_name1}.npy
│   ├── {video_name2}.npy
│   └── ...
└── egoexo4d/
    ├── {video_name3}.json
    ├── {video_name4}.json
    └── ...
```
### 视频去畸变 (Video Undistortion)
给定按 [数据下载](#2-数据下载) 中描述的结构组织的原始视频和提供的相机内参，可以使用以下脚本对鱼眼畸变视频进行去畸变：
```bash
cd data/preprocessing

# 针对 Ego4D 视频
usage: undistort_video.py [-h] --video_root VIDEO_ROOT --intrinsics_root INTRINSICS_ROOT --save_root SAVE_ROOT [--video_start START_IDX] [--video_END END_IDX] [--batchsize BATCHSIZE] [--crf CRF]

options:
  -h, --help                            显示此帮助信息并退出
  --video_root VIDEO_ROOT               包含输入视频的文件夹
  --intrinsics_root INTRINSICS_ROOT     包含内参信息的文件夹
  --save_root SAVE_ROOT                 保存输出视频的文件夹
  --video_start VIDEO_START             起始视频索引（包含）
  --video_end VIDEO_END                 结束视频索引（不包含）
  --batch_size BATCH_SIZE               每批处理的帧数（TS chunk）
  --crf CRF                             ffmpeg 编码质量的 CRF 值
```

示例命令如下：
```bash
# 针对 Ego4D 视频
python undistort_video.py --video_root Ego4D_root/v2/full_scale --intrinsics_root camera_intrinsics_root/ego4d --save_root Ego4D_undistorted_root --video_start 0 --video_end 10
```
这将按顺序处理 10 个 Ego4D 视频，并将去畸变后的输出保存到 `Ego4D_root/v2/undistorted_videos`。

同样，对于 EgoExo4D 视频，您可以运行如下命令：
```bash
# 针对 EgoEXO4D 视频
python undistort_video_egoexo4d.py --video_root EgoExo4D_root --intrinsics_root camera_intrinsics_root/egoexo4d --save_root EgoExo4D_undistorted_root --video_start 0 --video_end 10
```

每个视频会根据指定的批大小分段处理，然后再拼接起来。值得注意的是，处理整个数据集非常耗时且需要大量存储空间（约 10 TB）。此处提供的脚本仅作为基本参考示例。**我们建议在计算集群上运行之前对其进行并行化和优化。**

**去畸变步骤仅适用于 Ego4D 和 EgoExo4D 视频。Epic-Kitchen 和 Somethingsomething-v2 不需要去畸变，可以直接使用从官方源下载的文件。**

---

## 4. 元数据结构 (Metadata Structure)
每个 V-L-A 片段的元数据可以通过以下方式加载：
```python
import numpy as np

# 加载元数据字典
episode_info = np.load(f'{dataset_name1}_{video_name1}_ep_{000000}.npy', allow_pickle=True).item()

```
`episode_info` 的详细结构如下：
```
episode_info (dict)                                 # 单个 V-L-A 片段的元数据
├── 'video_clip_id_segment': list[int]              # 已弃用
├── 'extrinsics': np.ndarray                        # (Tx4x4) 世界坐标系到相机坐标系的外参矩阵 (World2Cam)
├── 'intrinsics': np.ndarray                        # (3x3) 相机内参矩阵
├── 'video_decode_frame': list[int]                 # 原始 raw 视频中的帧索引（从 0 开始）
├── 'video_name': str                               # 原始 raw 视频名称
├── 'avg_speed': float                              # 每帧及手腕的平均移动距离（米）
├── 'total_rotvec_degree': float                    # 片段内的总相机旋转（度）
├── 'total_transl_dist': float                      # 片段内的总相机平移距离（米）
├── 'anno_type': str                                # 标注类型，指定分割片段时主要考虑的手部动作
├── 'text': (dict)                                  # 片段的文本描述
│     ├── 'left': List[(str, (int, int))]           # 每个条目包含（描述，(片段内开始帧, 片段内结束帧)）
│     └── 'right': List[(str, (int, int))]          # 右手的相同结构
├── 'text_rephrase': (dict)                         # GPT-4 改写的文本描述
│     ├── 'left': List[(List[str], (int, int))]     # 每个条目包含（改写描述列表，(片段内开始帧, 片段内结束帧)）
│     └── 'right': List[(List[str], (int, int))]    # 右手的相同结构
├── 'left' (dict)                                   # 左手 3D 姿态信息
│   ├── 'beta': np.ndarray                          # (10) MANO 手部形状参数（基于 MANO_RIGHT 模型）
│   ├── 'global_orient_camspace': np.ndarray        # (Tx3x3) 从 MANO 规范空间到相机空间的手腕旋转
│   ├── 'global_orient_worldspace': np.ndarray      # (Tx3x3) 从 MANO 规范空间到世界空间的手腕旋转
│   ├── 'hand_pose': np.ndarray                     # (Tx15x3x3) 局部手关节旋转（基于 MANO_RIGHT 模型）
│   ├── 'transl_camspace': np.ndarray               # (Tx3) 已弃用
│   ├── 'transl_worldspace': np.ndarray             # (Tx3) 世界空间中的手腕平移
│   ├── 'kept_frames': list[int]                    # (T) 有效左手重建帧的 0-1 掩码
│   ├── 'joints_camspace': np.ndarray               # (Tx21x3) 相机空间中的 3D 手部关节位置
│   ├── 'joints_worldspace': np.ndarray             # (Tx21x3) 世界空间中的 3D 关节位置
│   ├── 'wrist': np.ndarray                         # 已弃用
│   ├── 'max_translation_movement': float           # 已弃用
│   ├── 'max_wrist_rotation_movement': float        # 已弃用
│   └── 'max_finger_joint_angle_movement': float    # 已弃用
└── 'right' (dict)                                  # 右手 3D 姿态信息（结构同 'left'）
    ├── 'beta': np.ndarray
    ├── 'global_orient_camspace': np.ndarray
    ├── 'global_orient_worldspace': np.ndarray
    ├── 'hand_pose': np.ndarray
    ├── 'transl_camspace': np.ndarray
    ├── 'transl_worldspace': np.ndarray
    ├── 'kept_frames': list[int]
    ├── 'joints_camspace': np.ndarray
    ├── 'joints_worldspace': np.ndarray
    ├── 'wrist': np.ndarray
    ├── 'max_translation_movement': float
    ├── 'max_wrist_rotation_movement': float
    └── 'max_finger_joint_angle_movement': float
```
为了更好地理解如何使用片段元数据，如下一节所述，我们提供了一个可视化脚本。

---

## 5. 数据可视化
每个片段的元数据可以使用以下命令进行可视化，该命令将生成与我们 [网页](https://microsoft.github.io/VITRA/) 上展示格式相同的视频。
我们建议按照上述说明进行去畸变处理，并将所有去畸变视频放在单个 `video_root` 文件夹中，将相应的元数据存储在 `label_root` 文件夹中，然后运行可视化脚本。

```bash
usage: data/demo_visualization_epi.py [-h] --video_root VIDEO_ROOT --label_root LABEL_ROOT --save_path SAVE_PATH --mano_model_path MANO_MODEL_PATH [--render_gradual_traj]

options:
  -h, --help                            显示此帮助信息并退出
  --video_root VIDEO_ROOT               包含视频文件的根目录
  --label_root LABEL_ROOT               包含片段标签 (.npy) 文件的根目录
  --save_path SAVE_PATH                 保存输出可视化视频的目录
  --mano_model_path MANO_MODEL_PATH     MANO 模型文件的路径
  --render_gradual_traj                 设置标志以渲染渐进轨迹（完整模式）
```
我们提供了一个运行脚本的示例命令，以及一个用于可视化的样本：
```bash
python data/demo_visualization_epi.py --video_root data/examples/videos --label_root data/examples/annotations --save_path data/examples/visualize --mano_model_path MANO_MODEL_PATH --render_gradual_traj
```
请注意，使用 `--render_gradual_traj` 会为每一帧渲染从当前帧到片段结束的手部轨迹，这可能会很慢。为了加快可视化速度，您可以省略此选项。


要更详细地了解元数据，请参阅 `visualization/visualize_core.py`。
