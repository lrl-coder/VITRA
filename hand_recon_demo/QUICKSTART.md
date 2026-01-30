# 快速开始指南 - 3D手部重建Demo

## 🎯 5分钟快速上手

### 步骤1: 检查环境

```bash
cd hand_recon_demo
python check_environment.py
```

如果所有检查通过，继续下一步。否则按提示解决问题。

### 步骤2: 准备输入数据

将你的手部视频放到合适的位置，例如：
```
./videos/hand_video.mp4
```

或者准备图像序列文件夹：
```
./images/hand_frames/
    frame_0001.jpg
    frame_0002.jpg
    ...
```

### 步骤3: 运行demo

**Windows用户：**
```bash
python demo.py --input ./videos/hand_video.mp4 --output ./output/result.mp4 --camera_fx 1000 --camera_fy 1000
```

**或直接运行批处理脚本（需要先编辑参数）：**
```bash
run_demo.bat
```

**Linux/Mac用户：**
```bash
chmod +x run_demo.sh
./run_demo.sh

### 步骤4: 查看结果

输出视频保存在指定路径（默认 `./output/hand_recon_result.mp4`）
手部位姿数据保存在指定路径（`.npy`文件）

## ❓ 常见问题快速解答

**Q: 不知道相机焦距怎么办？**
# 尝试使用图像宽度作为初始值
python demo.py --input video.mp4 --output result.mp4 --camera_fx 1920 --camera_fy 1920
```

**Q: 处理速度太慢？**
```bash
# 禁用3D渲染
python demo.py --input video.mp4 --output result.mp4 --camera_fx 1000 --camera_fy 1000 --no_3d
```

**Q: 内存不够？**
```bash
# 限制处理帧数
python demo.py --input video.mp4 --output result.mp4 --camera_fx 1000 --camera_fy 1000 --max_frames 100
```

**Q: 没有GPU？**
```bash
# 使用CPU模式
python demo.py --input video.mp4 --output result.mp4 --camera_fx 1000 --camera_fy 1000 --device cpu
```

## 📊 参数调优建议

### 检测阈值 (--thresh)
- **默认**: 0.5
- **高质量视频**: 0.6-0.7 （减少误检）
- **模糊视频**: 0.3-0.4 （增加检测率）

### 输出帧率 (--fps)
- **流畅播放**: 30 fps
- **慢动作分析**: 60 fps
- **快速预览**: 15 fps

### 可视化选项
- **仅2D关键点**: `--no_mesh --no_3d`
- **仅网格**: `--no_2d --no_3d`
- **最快速度**: `--no_mesh --no_3d`

## 🔗 更多信息

详细文档请参考: [README.md](README.md)
