#!/bin/bash

# 手部重建Demo运行示例脚本

echo "======================================"
echo "3D手部重建Demo - 运行示例"
echo "======================================"

# 配置参数
INPUT_VIDEO="./videos/hand_video.mp4"  # 输入视频路径
OUTPUT_VIDEO="./output/hand_recon_result.mp4"  # 输出视频路径
SAVE_POSE="./output/hand_pose.npy"  # 保存手部位姿数据路径（可选）

# 相机内参（根据实际情况修改）
CAMERA_FX=1000.0  # X方向焦距
CAMERA_FY=1000.0  # Y方向焦距
# CAMERA_CX=640.0  # 主点X坐标（可选，默认使用图像中心）
# CAMERA_CY=360.0  # 主点Y坐标（可选，默认使用图像中心）

# 处理参数
MAX_FRAMES=100    # 最大处理帧数（测试用，设为None处理全部）
THRESH=0.5        # 检测阈值
FPS=30            # 输出视频帧率
DEVICE="cuda"     # 运行设备（cuda或cpu）

# 创建输出目录
mkdir -p ./output

echo ""
echo "参数配置："
echo "  输入: $INPUT_VIDEO"
echo "  输出: $OUTPUT_VIDEO"
echo "  位姿数据: $SAVE_POSE"
echo "  焦距: fx=$CAMERA_FX, fy=$CAMERA_FY"
echo "  设备: $DEVICE"
echo ""

# 运行demo
python demo.py \
    --input "$INPUT_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --camera_fx $CAMERA_FX \
    --camera_fy $CAMERA_FY \
    --max_frames $MAX_FRAMES \
    --thresh $THRESH \
    --fps $FPS \
    --device $DEVICE \
    --save_pose "$SAVE_POSE"

echo ""
echo "======================================"
echo "处理完成！"
echo "输出视频: $OUTPUT_VIDEO"
echo "位姿数据: $SAVE_POSE"
echo "======================================"
