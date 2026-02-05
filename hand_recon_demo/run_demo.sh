#!/bin/bash

# ============================================
# 手部重建Demo - 批量处理脚本
# ============================================
# 功能：批量处理文件夹中的视频文件，自动匹配同名内参文件
# 修改路径：只需修改下方的"路径配置"部分
# ============================================

echo "======================================"
echo "3D手部重建Demo - 批量处理"
echo "======================================"

export CUDA_VISIBLE_DEVICES=2

# ============================================
# 路径配置（根据需要修改这里）
# ============================================
# 视频文件夹路径
VIDEO_DIR="./hand_recon_demo/intrinsics"

# Annotation文件夹路径（包含内参信息的npy文件）
ANNOTATIONS_DIR="./data/examples/annotations"

# Annotation文件匹配模式（{video_name}会被替换为视频文件名）
# 例如：视频文件 03cc49c3-a7d1-445b-9a2a-545c4fae6843.mp4
#      对应annotation文件 Ego4D_03cc49c3-a7d1-445b-9a2a-545c4fae6843_ep_example.npy
ANNOTATION_PATTERN="Ego4D_{video_name}_ep_example.npy"

# 输出文件夹路径
OUTPUT_DIR="./hand_recon_demo/output"

# ============================================
# 处理参数（根据需要调整）
# ============================================
# 支持的视频格式（以空格分隔）
VIDEO_FORMATS="mp4 webm avi mov mkv"

# 处理参数
MAX_FRAMES=100    # 最大处理帧数（设为None处理全部）
THRESH=0.5        # 检测阈值
DEVICE="cuda"     # 运行设备（cuda或cpu）

# ============================================
# 批量处理开始
# ============================================
echo ""
echo "配置信息："
echo "  视频目录: $VIDEO_DIR"
echo "  Annotation目录: $ANNOTATIONS_DIR"
echo "  Annotation模式: $ANNOTATION_PATTERN"
echo "  输出目录: $OUTPUT_DIR"
echo "  设备: $DEVICE"
echo "  最大帧数: $MAX_FRAMES"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 函数：从annotation npy文件读取相机参数
read_intrinsics() {
    local annotation_file="$1"
    
    if [ ! -f "$annotation_file" ]; then
        echo "错误: annotation文件不存在: $annotation_file"
        return 1
    fi
    
    # 创建临时Python脚本
    local temp_script=$(mktemp /tmp/read_intrinsics_XXXXXX.py)
    
    cat > "$temp_script" << 'PYTHON_SCRIPT'
import numpy as np
import sys

annotation_file = sys.argv[1]

try:
    # 加载annotation npy文件
    data = np.load(annotation_file, allow_pickle=True).item()
    
    # 从'intrinsics'字段获取相机内参矩阵
    if 'intrinsics' not in data:
        print('ERROR: Cannot find intrinsics field in annotation file', file=sys.stderr)
        sys.exit(1)
    
    intrinsics = np.array(data['intrinsics'])
    
    # 从相机矩阵提取参数: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    
    # 输出参数（以空格分隔）
    print(f'{fx} {fy} {cx} {cy}')
    
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
    
    # 执行Python脚本
    local params=$(python "$temp_script" "$annotation_file" 2>&1)
    local exit_code=$?
    
    # 删除临时文件
    rm -f "$temp_script"
    
    # 检查Python脚本是否成功执行
    if [ $exit_code -ne 0 ] || [[ "$params" == ERROR:* ]]; then
        echo "$params"
        return 1
    fi
    
    # 解析参数
    read CAMERA_FX CAMERA_FY CAMERA_CX CAMERA_CY <<< "$params"
    
    # 检查是否成功提取
    if [ -z "$CAMERA_FX" ] || [ -z "$CAMERA_FY" ] || [ -z "$CAMERA_CX" ] || [ -z "$CAMERA_CY" ]; then
        echo "错误: 无法从annotation文件解析参数: $annotation_file"
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
        
        # 根据模式查找对应的annotation文件
        annotation_filename="${ANNOTATION_PATTERN/\{video_name\}/$filename}"
        annotation_file="$ANNOTATIONS_DIR/$annotation_filename"
        
        if [ ! -f "$annotation_file" ]; then
            echo "警告: 未找到对应的annotation文件: $annotation_file"
            echo "跳过此文件..."
            failed_files=$((failed_files + 1))
            continue
        fi
        
        # 读取内参
        if ! read_intrinsics "$annotation_file"; then
            echo "跳过此文件..."
            failed_files=$((failed_files + 1))
            continue
        fi
        
        echo "  视频文件: $video_file"
        echo "  Annotation文件: $annotation_file"
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
