#!/bin/bash

# 手部重建Demo运行示例脚本

echo "======================================"
echo "3D手部重建Demo - 运行示例"
echo "======================================"

export CUDA_VISIBLE_DEVICES=2

# 配置参数
# 待处理的视频文件名
FILE_NAME="102185"
# 输入视频路径
INPUT_DIR="./hand_recon_demo/test_data/video"
INPUT_VIDEO="${INPUT_DIR}/${FILE_NAME}.webm"
# 输出视频、手部位姿数据路径
OUTPUT_DIR="./hand_recon_demo/output"
OUTPUT_VIDEO="${OUTPUT_DIR}/vis_${FILE_NAME}.mp4"
SAVE_POSE="${OUTPUT_DIR}/hand_pose_${FILE_NAME}.npy"

# 相机内参
CAMERA_FX=358.82147216796875  # X方向焦距
CAMERA_FY=358.82147216796875  # Y方向焦距
CAMERA_CX=213.0  # 主点X坐标
CAMERA_CY=120.0  # 主点Y坐标

# 处理参数
MAX_FRAMES=100    # 最大处理帧数（测试用，设为None处理全部）
THRESH=0.5        # 检测阈值
DEVICE="cuda"     # 运行设备（cuda或cpu）

# 创建输出目录
mkdir -p $OUTPUT_DIR


echo ""
echo "参数配置："
echo "  输入: $INPUT_VIDEO"
echo "  输出: $OUTPUT_VIDEO"
echo "  位姿数据: $SAVE_POSE"
echo "  焦距: fx=$CAMERA_FX, fy=$CAMERA_FY"
echo "  主点: cx=$CAMERA_CX, cy=$CAMERA_CY"
echo "  设备: $DEVICE"
echo ""

# 运行demo
python hand_recon_demo/demo.py \
    --input "$INPUT_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --camera_fx $CAMERA_FX \
    --camera_fy $CAMERA_FY \
    --camera_cx $CAMERA_CX \
    --camera_cy $CAMERA_CY \
    --max_frames $MAX_FRAMES \
    --thresh $THRESH \
    --device $DEVICE \
    --save_pose "$SAVE_POSE"

echo ""
echo "======================================"
echo "处理完成！"
echo "输出视频: $OUTPUT_VIDEO"
echo "位姿数据: $SAVE_POSE"
echo "======================================"
