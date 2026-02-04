"""
手部重建Demo主程序

演示如何使用已知相机内参进行手部重建，并将结果可视化输出为视频。

使用方法:
    python demo.py --input <输入视频或图像序列> --output <输出视频路径> --camera_fx <焦距x> --camera_fy <焦距y>

示例:
    python demo.py --input ./videos/hand_video.mp4 --output ./output/result.mp4 --camera_fx 1000 --camera_fy 1000
"""

import argparse
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from hand_recon_known_camera import HandReconstructorWithKnownCamera
from visualizer import HandVisualizer


def load_images_from_video(video_path: str, max_frames: int = None) -> tuple:
    """
    从视频文件加载图像序列
    
    参数:
        video_path: 视频文件路径
        max_frames: 最大加载帧数（None表示全部加载）
    
    返回:
        (图像列表, 原始视频帧率)
    """
    cap = cv2.VideoCapture(video_path)
    images = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原始视频帧率
    
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    
    print(f"从视频加载图像: {video_path}")
    print(f"总帧数: {frame_count}")
    print(f"原始帧率: {fps:.2f} fps")
    
    with tqdm(total=frame_count, desc="加载帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            images.append(frame)
            pbar.update(1)
            
            if max_frames is not None and len(images) >= max_frames:
                break
    
    cap.release()
    print(f"成功加载 {len(images)} 帧")
    
    return images, fps


def load_images_from_folder(folder_path: str, max_frames: int = None) -> list:
    """
    从文件夹加载图像序列
    
    参数:
        folder_path: 图像文件夹路径
        max_frames: 最大加载帧数
    
    返回:
        图像列表
    """
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in supported_formats:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    if max_frames is not None:
        image_files = image_files[:max_frames]
    
    print(f"从文件夹加载图像: {folder_path}")
    print(f"找到 {len(image_files)} 张图像")
    
    images = []
    for img_path in tqdm(image_files, desc="加载图像"):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    
    print(f"成功加载 {len(images)} 张图像")
    
    return images


def create_camera_intrinsics(
    fx: float,
    fy: float,
    cx: float = None,
    cy: float = None,
    image_width: int = None,
    image_height: int = None
) -> np.ndarray:
    """
    创建相机内参矩阵
    
    参数:
        fx: X方向焦距
        fy: Y方向焦距
        cx: 主点X坐标（None则使用图像中心）
        cy: 主点Y坐标（None则使用图像中心）
        image_width: 图像宽度
        image_height: 图像高度
    
    返回:
        3x3的相机内参矩阵K
    """
    if cx is None:
        if image_width is None:
            raise ValueError("必须提供 cx 或 image_width")
        cx = image_width / 2.0
    
    if cy is None:
        if image_height is None:
            raise ValueError("必须提供 cy 或 image_height")
        cy = image_height / 2.0
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    return K


def save_hand_pose(recon_results: dict, output_path: str):
    """
    保存手部位姿数据（与原始数据库字段名一致）
    
    保存内容：
        - beta: MANO形状参数
        - global_orient_camspace: 手腕旋转矩阵（相机坐标系）
        - hand_pose: 手指关节旋转矩阵
        - transl_camspace: 手腕3D位移（相机坐标系）
        - joints_camspace: 手部关节3D位置（相机坐标系）
    
    参数:
        recon_results: 重建结果字典，包含 'left' 和 'right' 手部数据
        output_path: 输出文件路径（.npy格式）
    """
    # 准备保存的数据结构（字段名与data.md中的原始数据库一致）
    pose_data = {
        'left': {},
        'right': {},
        'description': {
            'beta': 'MANO形状参数 (10,)',
            'global_orient_camspace': '手腕旋转矩阵 (Tx3x3) - 从MANO规范空间到相机空间',
            'hand_pose': '手指关节旋转矩阵 (Tx15x3x3) - 局部手关节旋转',
            'transl_camspace': '手腕3D位移 (Tx3) - 相机坐标系',
            'joints_camspace': '手部关节3D位置 (Tx21x3) - 相机坐标系',
            'note': '字段名与原始数据库保持一致（参见data/data.md）',
            'usage': '顶点计算公式: V_cam = global_orient_camspace @ (MANO(beta, hand_pose) - wrist) + transl_camspace'
        }
    }
    
    for hand_type in ['left', 'right']:
        hand_data = recon_results.get(hand_type, {})
        
        # 收集所有帧的数据
        all_global_orient = []
        all_hand_pose = []
        all_transl = []
        all_joints = []
        all_beta = []
        
        for frame_idx in sorted(hand_data.keys()):
            frame_data = hand_data[frame_idx]
            all_global_orient.append(frame_data['global_orient'])  # (3, 3)
            all_hand_pose.append(frame_data['hand_pose'])  # (15, 3, 3)
            all_transl.append(frame_data['transl'])  # (3,)
            all_beta.append(frame_data['beta'])  # (10,)
            
            # 如果有joints数据也保存
            if 'joints' in frame_data:
                all_joints.append(frame_data['joints'])  # (21, 3)
        
        # 转换为numpy数组并保存（字段名与数据库一致）
        if len(all_global_orient) > 0:
            # 计算beta参数的平均值（所有帧的平均）
            beta_mean = np.mean(all_beta, axis=0)  # (10,)
            
            pose_data[hand_type] = {
                'beta': beta_mean,  # (10,) - 所有帧的平均值
                'global_orient_camspace': np.array(all_global_orient),  # (T, 3, 3)
                'hand_pose': np.array(all_hand_pose),  # (T, 15, 3, 3)
                'transl_camspace': np.array(all_transl),  # (T, 3)
                'joints_camspace': np.array(all_joints)  # (T, 21, 3)
            }
            
    
    # 保存为.npy文件
    np.save(output_path, pose_data)
    
    # 统计信息
    left_frames = len(pose_data['left'].get('global_orient_camspace', [])) if 'beta' in pose_data['left'] else 0
    right_frames = len(pose_data['right'].get('global_orient_camspace', [])) if 'beta' in pose_data['right'] else 0
    
    print(f"\n手部位姿数据已保存到: {output_path}")
    print(f"  左手帧数: {left_frames}")
    print(f"  右手帧数: {right_frames}")
    print(f"\n保存的数据（字段名与原始数据库一致）:")
    print(f"  - beta: (10,) MANO形状参数")
    print(f"  - global_orient_camspace: (T, 3, 3) 手腕旋转矩阵")
    print(f"  - hand_pose: (T, 15, 3, 3) 手指关节旋转矩阵")
    print(f"  - transl_camspace: (T, 3) 手腕3D位移")
    if left_frames > 0 and 'joints_camspace' in pose_data['left']:
        print(f"  - joints_camspace: (T, 21, 3) 手部关节3D位置")
    print(f"\n加载数据示例:")
    print(f"  data = np.load('{output_path}', allow_pickle=True).item()")
    print(f"  left_hand = data['left']")
    print(f"  beta = left_hand['beta']  # shape: (10,)")
    print(f"  global_orient = left_hand['global_orient_camspace']  # shape: (T, 3, 3)")
    print(f"  hand_pose = left_hand['hand_pose']  # shape: (T, 15, 3, 3)")
    print(f"  transl = left_hand['transl_camspace']  # shape: (T, 3)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用已知相机内参进行3D手部重建"
    )
    
    # 输入输出参数
    parser.add_argument('--input', type=str, required=True,
                        help='输入视频文件或图像文件夹路径')
    parser.add_argument('--output', type=str, default='./output/hand_recon_result.mp4',
                        help='输出视频路径')
    
    # 相机参数
    parser.add_argument('--camera_fx', type=float, required=True,
                        help='相机焦距 fx')
    parser.add_argument('--camera_fy', type=float, required=True,
                        help='相机焦距 fy')
    parser.add_argument('--camera_cx', type=float, default=None,
                        help='相机主点 cx（默认使用图像中心）')
    parser.add_argument('--camera_cy', type=float, default=None,
                        help='相机主点 cy（默认使用图像中心）')
    
    # 模型路径参数
    parser.add_argument('--hawor_model', type=str,
                        default='./weights/hawor/checkpoints/hawor.ckpt',
                        help='HaWoR模型路径')
    parser.add_argument('--detector', type=str,
                        default='./weights/hawor/external/detector.pt',
                        help='手部检测器路径')
    parser.add_argument('--mano_path', type=str,
                        default='./weights/mano',
                        help='MANO模型路径')
    
    # 处理参数
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最大处理帧数（用于测试）')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='手部检测置信度阈值')
    parser.add_argument('--fps', type=int, default=None,
                        help='输出视频帧率（默认使用原始视频帧率，图像序列默认为30）')
    
    # 保存参数
    parser.add_argument('--save_pose', type=str, default=None,
                        help='保存手部位姿数据的路径（.npy格式），不指定则不保存')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='运行设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载输入图像
    print("\n" + "="*60)
    print("步骤 1: 加载输入图像")
    print("="*60)
    
    input_path = Path(args.input)
    source_fps = None  # 原始视频的帧率
    
    if input_path.is_file():
        # 视频文件
        images, source_fps = load_images_from_video(str(input_path), args.max_frames)
    elif input_path.is_dir():
        # 图像文件夹
        images = load_images_from_folder(str(input_path), args.max_frames)
        source_fps = 30  # 图像序列默认帧率
    else:
        raise ValueError(f"输入路径不存在: {args.input}")
    
    if len(images) == 0:
        print("错误：没有加载到图像")
        return
    
    H, W = images[0].shape[:2]
    print(f"图像尺寸: {W}x{H}")
    
    # 确定输出视频的帧率
    if args.fps is not None:
        output_fps = args.fps
        print(f"使用指定帧率: {output_fps} fps")
    else:
        output_fps = source_fps if source_fps is not None else 30
        print(f"使用原始帧率: {output_fps} fps")
    
    # 2. 创建相机内参矩阵
    print("\n" + "="*60)
    print("步骤 2: 创建相机内参矩阵")
    print("="*60)
    
    camera_intrinsics = create_camera_intrinsics(
        fx=args.camera_fx,
        fy=args.camera_fy,
        cx=args.camera_cx,
        cy=args.camera_cy,
        image_width=W,
        image_height=H
    )
    
    print("相机内参矩阵 K:")
    print(camera_intrinsics)
    
    # 3. 初始化手部重建器
    print("\n" + "="*60)
    print("步骤 3: 初始化手部重建器")
    print("="*60)
    
    reconstructor = HandReconstructorWithKnownCamera(
        hawor_model_path=args.hawor_model,
        detector_path=args.detector,
        mano_path=args.mano_path,
        device=args.device
    )
    
    print("手部重建器初始化完成")
    
    # 4. 执行手部重建
    print("\n" + "="*60)
    print("步骤 4: 执行手部重建")
    print("="*60)
    
    recon_results = reconstructor.recon(
        images=images,
        camera_intrinsics=camera_intrinsics,
        thresh=args.thresh
    )
    
    # 统计检测结果
    left_count = len(recon_results['left'])
    right_count = len(recon_results['right'])
    print(f"检测到左手: {left_count} 帧")
    print(f"检测到右手: {right_count} 帧")
    
    # 5. 可视化并生成视频
    print("\n" + "="*60)
    print("步骤 5: 生成可视化视频")
    print("="*60)
    
    visualizer = HandVisualizer()
    
    visualizer.create_video_with_3d_hands(
        images=images,
        recon_results=recon_results,
        camera_intrinsics=camera_intrinsics,
        output_path=args.output,
        fps=output_fps
    )
    
    # 6. 保存手部位姿数据（可选）
    if args.save_pose is not None:
        print("\n" + "="*60)
        print("步骤 6: 保存手部位姿数据")
        print("="*60)
        save_hand_pose(recon_results, args.save_pose)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print(f"输出视频: {args.output}")
    if args.save_pose is not None:
        print(f"位姿数据: {args.save_pose}")


if __name__ == "__main__":
    main()
