# 3D手部重建Demo（已知相机内参版本）

这是一个基于已知相机内参的3D手部重建演示程序，可以从视频或图像序列中重建手部的3D模型，并生成可视化视频。

## 📋 主要功能

- ✅ 使用已知相机内参进行3D手部重建（无需MoGe估计）
- ✅ 支持左右手同时检测和重建
- ✅ 2D关键点可视化
- ✅ 3D网格线框可视化
- ✅ 3D手部渲染视图
- ✅ 输出可视化视频

## 🏗️ 项目结构

```
hand_recon_demo/
├── hand_recon_known_camera.py  # 手部重建核心模块
├── visualizer.py               # 可视化模块
├── demo.py                     # 主程序
├── requirements.txt            # 依赖列表
├── README.md                   # 说明文档
└── examples/                   # 示例脚本
    └── run_demo.sh            # 运行示例
```

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
    --fps 30 \
    --no_3d
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

### 可视化控制参数

| 参数 | 说明 |
|------|------|
| `--no_2d` | 不绘制2D关键点 |
| `--no_mesh` | 不绘制网格线框 |
| `--no_3d` | 不渲染3D视图（可加快处理速度） |

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

程序会生成包含以下可视化内容的视频：

1. **左侧视图**：原始图像 + 2D关键点 + 网格线框
2. **右侧视图**（如果启用3D渲染）：3D手部模型渲染图

颜色编码：
- 🟢 **绿色**：左手
- 🔴 **红色**：右手

## 🔧 技术细节

### 处理流程

1. **图像加载**：从视频或图像文件夹加载图像序列
2. **手部检测**：使用检测器定位图像中的手部区域
3. **姿态估计**：使用HaWoR估计手部的姿态、形状和位移参数
4. **MANO建模**：使用MANO模型生成3D手部网格
5. **坐标对齐**：将手部坐标对齐到全局坐标系
6. **可视化**：生成2D/3D可视化并输出视频

### 与 `hand_recon_core.py` 的区别

| 特性 | hand_recon_core.py | 本Demo |
|------|-------------------|--------|
| 相机参数估计 | 使用MoGe自动估计FOV | 使用已知内参，无需估计 |
| 适用场景 | 未知相机参数 | 已知相机参数 |
| 处理速度 | 较慢（需要MoGe推理） | 较快 |
| 可视化 | 无 | 完整的2D/3D可视化 |

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
- 使用 `--no_3d` 禁用3D渲染（最慢的部分）
- 使用 `--no_mesh` 禁用网格绘制
- 使用 GPU 加速：`--device cuda`

## 📝 许可证

本项目基于VITRA项目，遵循相同的许可证。

## 🙏 致谢

- **HaWoR**：手部姿态估计
- **MANO**：参数化手部模型
- **MoGe**：相机参数估计（原版使用）

## 📧 联系方式

如有问题，请联系项目维护者。
