# DROID-SLAM 安装指南

本指南介绍如何在 VITRA 项目中安装 DROID-SLAM 用于相机位姿估计。

## 系统要求

- **GPU**: 至少 11GB 显存（推理）
- **CUDA**: 11.3+ (推荐)
- **Python**: 3.8+
- **操作系统**: Linux (推荐) / Windows (需要 Visual Studio)

## 安装步骤

### 方法一：在现有 VITRA 环境中安装（推荐）

如果你已经有 VITRA 环境，可以直接在其中安装 DROID-SLAM 的依赖。

#### 1. 安装必要依赖

```bash
# 进入 VITRA 项目根目录
cd d:\project\PYProject\VITRA

# 激活你的 conda 环境
conda activate vitra  # 或你的环境名称

# 安装 pytorch-scatter (需要匹配你的 PyTorch 版本)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 安装 lietorch 依赖
pip install evo --upgrade --no-binary evo
pip install gdown
```

#### 2. 编译 DROID-SLAM 扩展

```bash
# 进入 DROID-SLAM 目录
cd thirdparty\HaWoR\thirdparty\DROID-SLAM

# 编译 C++/CUDA 扩展（需要约 10 分钟）
python setup.py install
```

**Windows 用户注意**：需要安装 Visual Studio Build Tools (C++ 工具链)

#### 3. 编译 lietorch

```bash
# 进入 lietorch 目录
cd thirdparty\lietorch

# 编译 lietorch
python setup.py install
```

#### 4. 下载预训练模型

```bash
# 回到 DROID-SLAM 目录
cd ..\..

# 下载模型权重 (约 500MB)
# 选项 1: 手动下载
# 访问 https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view
# 保存到 weights/external/droid.pth

# 选项 2: 使用 gdown
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O weights/external/droid.pth
```

### 方法二：创建独立的 DROID-SLAM 环境

如果你遇到依赖冲突，可以创建独立环境：

```bash
cd thirdparty\HaWoR\thirdparty\DROID-SLAM

# 使用无可视化的环境配置（更轻量）
conda env create -f environment_novis.yaml

# 或完整环境
conda env create -f environment.yaml

# 激活环境
conda activate droidenv

# 安装额外依赖
pip install evo --upgrade --no-binary evo
pip install gdown

# 编译扩展
python setup.py install

# 编译 lietorch
cd thirdparty\lietorch
python setup.py install
```

## 验证安装

```python
# 测试导入
python -c "from droid import Droid; print('DROID-SLAM installed successfully!')"
```

## 常见问题

### 1. 编译错误：找不到 CUDA

确保 CUDA 工具包已正确安装：
```bash
nvcc --version
```

如果没有，安装 CUDA Toolkit：
- 官网: https://developer.nvidia.com/cuda-downloads

### 2. Windows 编译错误

需要安装 Visual Studio Build Tools：
1. 下载 Visual Studio Installer
2. 安装 "Desktop development with C++"
3. 包含 "MSVC" 和 "Windows 10/11 SDK"

### 3. torch-scatter 安装失败

尝试从源码安装：
```bash
pip install git+https://github.com/rusty1s/pytorch_scatter.git
```

### 4. 显存不足

DROID-SLAM 需要至少 11GB 显存。如果显存不足：
- 降低输入图像分辨率
- 增加 `stride` 参数（跳帧处理）
- 使用 `filter_thresh` 参数过滤关键帧

## 不安装 DROID-SLAM 的替代方案

如果安装 DROID-SLAM 困难，VITRA pipeline 会自动回退到 **静态相机假设**：

- `extrinsics` 使用单位矩阵
- `camera-space` = `world-space`
- 适用于三脚架拍摄的稳定视频

这种模式适用于：
- 相机基本不动的视频
- 机器人遥操作演示视频（固定视角）

## 模型权重位置

确保权重文件放在正确位置：
```
VITRA/
└── weights/
    └── external/
        └── droid.pth
```
