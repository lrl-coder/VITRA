#!/bin/bash

# ============================================
# 批量手部重建Demo运行脚本
# ============================================
# 功能：批量处理文件夹中的视频文件，自动匹配同名内参文件
# 
# 使用说明：
# 1. 修改下方的"路径配置"部分，指定视频和内参的文件夹路径
# 2. 根据需要调整"处理参数"部分
# 3. 运行脚本：bash run_batch_demo.sh
# ============================================

# ============================================
# 路径配置（根据需要修改）
# ============================================
# 视频文件夹路径
VIDEO_DIR="./hand_recon_demo/intrinsics"

# 内参文件夹路径（如果和视频在同一文件夹，设置为和VIDEO_DIR相同）
INTRINSICS_DIR="./hand_recon_demo/intrinsics"

# 输出文件夹路径
OUTPUT_DIR="./hand_recon_demo/output"

# ============================================
# 处理参数（根据需要调整）
# ============================================
# CUDA设备
export CUDA_VISIBLE_DEVICES=2

# 支持的视频格式（以空格分隔）
VIDEO_FORMATS="mp4 webm avi mov mkv"

# 处理参数
MAX_FRAMES=100    # 最大处理帧数（设为None处理全部）
THRESH=0.5        # 检测阈值
DEVICE="cuda"     # 运行设备（cuda或cpu）

# ============================================
# 脚本开始
# ============================================
echo "======================================"
echo "3D手部重建批量处理Demo"
echo "======================================"
echo ""
echo "配置信息："
echo "  视频目录: $VIDEO_DIR"
echo "  内参目录: $INTRINSICS_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  设备: $DEVICE"
echo "  最大帧数: $MAX_FRAMES"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 函数：从内参文件读取相机参数
read_intrinsics() {
    local intrinsic_file="$1"
    
    if [ ! -f "$intrinsic_file" ]; then
        echo "错误: 内参文件不存在: $intrinsic_file"
        return 1
    fi
    
    # 读取内参文件（格式: fx = 435.36, fy = 435.36, ppx = 213.50, ppy = 120.00）
    local content=$(cat "$intrinsic_file")
    
    # 使用sed提取参数
    CAMERA_FX=$(echo "$content" | sed -n 's/.*fx = \([0-9.]*\).*/\1/p')
    CAMERA_FY=$(echo "$content" | sed -n 's/.*fy = \([0-9.]*\).*/\1/p')
    CAMERA_CX=$(echo "$content" | sed -n 's/.*ppx = \([0-9.]*\).*/\1/p')
    CAMERA_CY=$(echo "$content" | sed -n 's/.*ppy = \([0-9.]*\).*/\1/p')
    
    # 检查是否成功提取
    if [ -z "$CAMERA_FX" ] || [ -z "$CAMERA_FY" ] || [ -z "$CAMERA_CX" ] || [ -z "$CAMERA_CY" ]; then
        echo "错误: 无法从内参文件解析参数: $intrinsic_file"
        return 1
    fi
    
    return 0
}

# 统计变量
total_files=0
processed_files=0
failed_files=0

# 遍历所有支持的视频格式
for ext in $VIDEO_FORMATS; do
    # 查找所有该格式的视频文件
    for video_file in "$VIDEO_DIR"/*.$ext; do
        # 检查文件是否存在（防止通配符没有匹配到文件的情况）
        [ -e "$video_file" ] || continue
        
        total_files=$((total_files + 1))
        
        # 获取文件名（不含扩展名）
        basename=$(basename "$video_file")
        filename="${basename%.*}"
        
        echo ""
        echo "======================================"
        echo "[$total_files] 处理文件: $filename"
        echo "======================================"
        
        # 查找对应的内参文件
        intrinsic_file="$INTRINSICS_DIR/${filename}.txt"
        
        if [ ! -f "$intrinsic_file" ]; then
            echo "警告: 未找到对应的内参文件: $intrinsic_file"
            echo "跳过此文件..."
            failed_files=$((failed_files + 1))
            continue
        fi
        
        # 读取内参
        if ! read_intrinsics "$intrinsic_file"; then
            echo "跳过此文件..."
            failed_files=$((failed_files + 1))
            continue
        fi
        
        echo "  视频文件: $video_file"
        echo "  内参文件: $intrinsic_file"
        echo "  相机参数: fx=$CAMERA_FX, fy=$CAMERA_FY, cx=$CAMERA_CX, cy=$CAMERA_CY"
        
        # 定义输出路径
        output_video="$OUTPUT_DIR/vis_${filename}.mp4"
        save_pose="$OUTPUT_DIR/hand_pose_${filename}.npy"
        
        echo "  输出视频: $output_video"
        echo "  位姿数据: $save_pose"
        echo ""
        
        # 运行demo
        python hand_recon_demo/demo.py \
            --input "$video_file" \
            --output "$output_video" \
            --camera_fx $CAMERA_FX \
            --camera_fy $CAMERA_FY \
            --camera_cx $CAMERA_CX \
            --camera_cy $CAMERA_CY \
            --max_frames $MAX_FRAMES \
            --thresh $THRESH \
            --device $DEVICE \
            --save_pose "$save_pose"
        
        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo "✓ 成功处理: $filename"
            processed_files=$((processed_files + 1))
        else
            echo "✗ 处理失败: $filename"
            failed_files=$((failed_files + 1))
        fi
    done
done

# 打印总结
echo ""
echo "======================================"
echo "批量处理完成！"
echo "======================================"
echo "总文件数: $total_files"
echo "成功处理: $processed_files"
echo "失败/跳过: $failed_files"
echo "输出目录: $OUTPUT_DIR"
echo "======================================"
