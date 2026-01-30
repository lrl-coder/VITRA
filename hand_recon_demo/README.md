# 3D手部重建Demo（已知相机内参版本）

这是一个基于已知相机内参的3D手部重建演示程序，可以从视频或图像序列中重建手部的3D模型，并生成可视化视频。

## 📋 主要功能

- ✅ 使用已知相机内参进行3D手部重建（无需MoGe估计）
- ✅ 支持左右手同时检测和重建
- ✅ 高质量3D手部网格渲染（基于PyTorch3D）
- ✅ 输出可视化视频
- ✅ 保存手部位姿数据（.npy格式）

## 🏗️ 项目结构

```
hand_recon_demo/
├── hand_recon_known_camera.py  # 手部重建核心模块
├── visualizer.py               # 可视化模块（基于PyTorch3D）
├── demo.py                     # 主程序
├── load_hand_pose.py           # 位姿数据加载工具
├── requirements.txt            # 依赖列表
├── check_environment.py        # 环境检查脚本
├── README.md                   # 使用说明文档
├── TECHNICAL_OVERVIEW.md       # 技术概述与实现详解
├── QUICKSTART.md               # 快速开始指南
└── run_demo.sh / .bat          # 运行示例脚本
```

📖 **推荐阅读顺序**：
1. `README.md` - 了解基本用法
2. `QUICKSTART.md` - 快速上手
3. `TECHNICAL_OVERVIEW.md` - 深入了解技术原理与实现细节

## 📦 依赖安装

### 基础依赖

```bash
pip install -r requirements.txt
```

### 必需的模型权重

确保以下模型权重已下载并放置在正确位置：

1. **HaWoR模型** (`./weights/hawor/checkpoints/hawor.ckpt`)
   - 用于手部姿态估计

2. **手部检测器** (`./weights/hawor/external/detector.pt`)
   - 用于检测图像中的手部区域

3. **MANO模型** (`./weights/mano/`)
   - 用于生成手部3D网格

## 🚀 快速开始

### 基本用法

```bash
python demo.py \
    --input <输入视频或图像文件夹> \
    --output <输出视频路径> \
    --camera_fx <焦距fx> \
    --camera_fy <焦距fy>
```

### 示例1：处理视频文件

```bash
python demo.py \
    --input ./videos/hand_video.mp4 \
    --output ./output/result.mp4 \
    --camera_fx 1000.0 \
    --camera_fy 1000.0
```

### 示例2：处理图像序列

```bash
python demo.py \
    --input ./images/hand_sequence/ \
    --output ./output/result.mp4 \
    --camera_fx 800.0 \
    --camera_fy 800.0 \
    --camera_cx 640.0 \
    --camera_cy 360.0
```

### 示例3：自定义参数

```bash
python demo.py \
    --input ./videos/hand_video.mp4 \
    --output ./output/result.mp4 \
    --camera_fx 1000.0 \
    --camera_fy 1000.0 \
    --max_frames 100 \
    --thresh 0.6 \
    --fps 30
```

## 🎛️ 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input` | 输入视频文件或图像文件夹路径 | `./videos/hand.mp4` |
| `--camera_fx` | 相机焦距 fx（像素单位） | `1000.0` |
| `--camera_fy` | 相机焦距 fy（像素单位） | `1000.0` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output` | 输出视频路径 | `./output/hand_recon_result.mp4` |
| `--camera_cx` | 相机主点 cx（默认使用图像中心） | `None` |
| `--camera_cy` | 相机主点 cy（默认使用图像中心） | `None` |
| `--max_frames` | 最大处理帧数（用于测试） | `None`（全部） |
| `--thresh` | 手部检测置信度阈值 | `0.5` |
| `--fps` | 输出视频帧率 | `30` |
| `--device` | 运行设备（cuda/cpu） | `cuda` |
| `--save_pose` | 保存手部位姿数据的路径（.npy格式） | `None`（不保存） |


### 模型路径参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hawor_model` | HaWoR模型路径 | `./weights/hawor/checkpoints/hawor.ckpt` |
| `--detector` | 手部检测器路径 | `./weights/hawor/external/detector.pt` |
| `--mano_path` | MANO模型路径 | `./weights/mano` |

## 📐 相机内参说明

相机内参矩阵 K 的格式为：

```
K = [fx,  0, cx]
    [ 0, fy, cy]
    [ 0,  0,  1]
```

其中：
- **fx, fy**: 焦距（单位：像素）
- **cx, cy**: 主点坐标（图像坐标系的原点，通常在图像中心）

### 如何获取相机内参？

1. **已知相机标定结果**：直接使用标定得到的内参
2. **已知FOV**：可以通过FOV计算焦距
   ```python
   fx = image_width / (2 * tan(fov_x / 2))
   fy = image_height / (2 * tan(fov_y / 2))
   ```
3. **估算值**：一般手机相机的焦距约为图像宽度的 0.8-1.2 倍

## 🎨 输出说明

程序会生成包含3D手部渲染的可视化视频。

### 可视化内容

- **3D手部网格**：使用PyTorch3D渲染的高质量3D手部模型
- **双手支持**：同时显示左右手（如果检测到）

颜色编码：
- 🟢 **绿色**：左手
- 🔴 **红色**：右手

## 🔧 技术细节

### 处理流程

1. **图像加载**：从视频或图像文件夹加载图像序列
2. **手部检测**：使用YOLOv8检测器定位图像中的手部区域
3. **姿态估计**：使用HaWoR估计手部的姿态、形状和位移参数
4. **MANO建模**：使用MANO模型生成3D手部网格（778个顶点，21个关节）
5. **坐标对齐**：将手部坐标对齐到相机坐标系
6. **PyTorch3D渲染**：生成高质量3D可视化并输出视频

### 核心技术

| 组件 | 技术 | 说明 |
|------|------|------|
| 手部检测 | YOLOv8 | 定位左右手区域 |
| 姿态估计 | HaWoR | 估计MANO参数 |
| 3D建模 | MANO | 参数化手部模型 |
| 渲染 | PyTorch3D | 专业级3D渲染 |

## 💾 保存的位姿数据结构

使用 `--save_pose output.npy` 可以保存手部位姿数据供后续使用。

### 数据格式

```python
{
    'left': {
        frame_idx: {
            'wrist_position': np.ndarray,    # (3,) 手腕3D位置 [x, y, z]
            'wrist_rotation': np.ndarray,    # (3, 3) 手腕旋转矩阵
            'finger_rotations': np.ndarray,  # (15, 3, 3) 手指关节旋转矩阵
            'shape_params': np.ndarray,      # (10,) MANO形状参数
        },
        ...  # 多帧数据
    },
    'right': {
        frame_idx: {...},  # 与left结构相同
        ...
    },
    'description': {
        'wrist_position': '手腕3D位置 (3,) - [x, y, z] 在相机坐标系中',
        'wrist_rotation': '手腕旋转矩阵 (3, 3) - global_orient',
        'finger_rotations': '15个手指关节的旋转矩阵 (15, 3, 3) - hand_pose',
        'shape_params': 'MANO形状参数 (10,) - beta',
        'usage': '顶点计算公式: V_cam = global_orient @ (MANO(beta, hand_pose) - wrist) + transl'
    }
}
```

### 参数说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `wrist_position` | `(3,)` | 手腕在相机坐标系下的3D位置 [x, y, z]（米） |
| `wrist_rotation` | `(3, 3)` | 手腕的全局旋转矩阵（global_orient） |
| `finger_rotations` | `(15, 3, 3)` | 15个手指关节的局部旋转矩阵 |
| `shape_params` | `(10,)` | MANO PCA形状参数（控制手部大小、粗细） |

#### 关节索引说明

15个手指关节对应：
- **拇指**：关节 0-2（3个）
- **食指**：关节 3-5（3个）
- **中指**：关节 6-8（3个）
- **无名指**：关节 9-11（3个）
- **小指**：关节 12-14（3个）

### 加载和使用示例

```python
import numpy as np

# 加载数据
data = np.load('hand_pose.npy', allow_pickle=True).item()

# 访问左手第0帧数据
left_frame_0 = data['left'][0]

# 提取参数
wrist_pos = left_frame_0['wrist_position']      # (3,)
wrist_rot = left_frame_0['wrist_rotation']      # (3, 3)
finger_rot = left_frame_0['finger_rotations']   # (15, 3, 3)
shape = left_frame_0['shape_params']            # (10,)

print(f"手腕位置: {wrist_pos}")
print(f"形状参数: {shape}")

# 使用MANO重建手部网格
from libs.models.mano_wrapper import MANO
import torch

mano = MANO(model_path='./weights/mano')
output = mano(
    betas=torch.tensor(shape).unsqueeze(0),
    hand_pose=torch.tensor(finger_rot).unsqueeze(0)
)

vertices = output.vertices[0]  # (778, 3) 手部顶点
joints = output.joints[0]      # (21, 3) 手部关节

# 应用全局变换到相机坐标系
wrist_joint = joints[0]
vertices_cam = (wrist_rot @ (vertices - wrist_joint).T).T + wrist_pos
```

### 坐标系说明

**相机坐标系**（右手系）：
- **X轴**：向右
- **Y轴**：向下
- **Z轴**：垂直于图像平面向前（深度方向）

所有保存的3D坐标都在相机坐标系中。

## 🐛 常见问题

### Q1: 运行时报错 "CUDA out of memory"

**解决方案**：
- 使用 `--device cpu` 切换到CPU模式
- 减少 `--max_frames` 限制处理帧数
- 使用较小分辨率的输入视频

### Q2: 没有检测到手部

**解决方案**：
- 降低检测阈值：`--thresh 0.3`
- 确保输入图像中手部清晰可见
- 检查手部检测器模型是否正确加载

### Q3: 焦距参数不知道怎么设置

**解决方案**：
- 如果有相机标定数据，直接使用标定结果
- 如果没有，可以尝试 `fx = fy = image_width` 作为初始值
- 根据可视化结果调整（关键点应该对齐到手部）

### Q4: 生成视频很慢

**解决方案**：
- 使用 GPU 加速：`--device cuda`
- 减少处理帧数：`--max_frames 100`
- 确保 PyTorch3D 已正确安装并支持 CUDA

## 📝 许可证

本项目基于VITRA项目，遵循相同的许可证。

## 🙏 致谢

- **HaWoR**：手部姿态估计
- **MANO**：参数化手部模型
- **MoGe**：相机参数估计（原版使用）

## 📧 联系方式

如有问题，请联系项目维护者。
