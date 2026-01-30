"""
3D手部可视化模块 (基于 PyTorch3D)

复用 VITRA 项目的 visualization.visualize_core 进行高质量渲染。
"""

import sys
import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径以便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入必要的模块，如果失败则给出提示
try:
    from visualization.visualize_core import HandVisualizer as BaseHandVisualizer, Config as BaseConfig
    from visualization.render_utils import Renderer
except ImportError as e:
    raise ImportError(f"无法导入 VITRA 可视化模块: {e}\n请确保在 VITRA 项目根目录下运行或正确设置了 PYTHONPATH。")

class DemoConfig(BaseConfig):
    """
    Demo 专用配置类，继承自基础配置
    """
    def __init__(self, mano_path='./weights/mano', fps=30):
        super().__init__()
        # 覆盖必要的路径和参数
        self.MANO_MODEL_PATH = mano_path
        self.FPS = fps
        
        # 保持默认颜色设置 (从基类继承)
        # self.LEFT_COLOR = ...
        # self.RIGHT_COLOR = ...

class HandVisualizer(BaseHandVisualizer):
    """
    适配 Demo 的手部可视化器
    """
    def __init__(self, mano_path='./weights/mano'):
        # 初始化配置
        config = DemoConfig(mano_path=mano_path)
        # 初始化基类
        # render_gradual_traj=False 对应仅基础模式
        super().__init__(config, render_gradual_traj=False)
        
        # [USER REQUEST]: 仅保留 'cam' 模式
        self.all_modes = ['cam']
        
    def create_video_with_3d_hands(
        self,
        images: List[np.ndarray],
        recon_results: Dict,
        camera_intrinsics: np.ndarray,
        output_path: str,
        fps: int = 30
    ):
        """
        生成带有3D手部渲染的可视化视频
        
        参数:
            images: 原始图像列表 (H, W, 3) uint8 BGR (opencv default)
            recon_results: 重建结果字典 {'left': {t: data}, ...}
            camera_intrinsics: 相机内参矩阵 (3, 3)
            output_path: 输出路径
            fps: 帧率
        """
        if not images:
            print("没有图像可处理")
            return

        print(f"开始生成高质量渲染视频 (PyTorch3D)...")
        
        self.config.FPS = fps
        T = len(images)
        H, W = images[0].shape[:2]
        
        # 1. 准备数据容器
        # Vertices shape: (T, 778, 3)
        verts_left_list = np.zeros((T, 778, 3), dtype=np.float32)
        verts_right_list = np.zeros((T, 778, 3), dtype=np.float32)
        mask_left = np.zeros(T, dtype=np.int32)
        mask_right = np.zeros(T, dtype=np.int32)
        
        # 2. 填充手部数据
        # 注意: 我们将直接使用 相机坐标系 作为 世界坐标系
        # 因此 Extrinsics 将设为单位阵
        # 参考 visualize_core.py 中的 process_single_hand_labels 函数
        
        # 初始化 MANO 模型
        from libs.models.mano_wrapper import MANO
        mano = MANO(model_path='./weights/mano').to('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for t in range(T):
            # 处理左手
            if t in recon_results.get('left', {}):
                res = recon_results['left'][t]
                
                # 提取参数
                beta = torch.from_numpy(res['beta']).unsqueeze(0).to(device)
                hand_pose = torch.from_numpy(res['hand_pose']).unsqueeze(0).to(device)
                global_orient = torch.from_numpy(res['global_orient']).unsqueeze(0).to(device)
                transl = torch.from_numpy(res['transl']).unsqueeze(0).to(device)
                
                # MANO前向传播（使用单位旋转作为占位符）
                identity_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).to(device)
                mano_out = mano(betas=beta, hand_pose=hand_pose, global_orient=identity_rot)
                verts = mano_out.vertices[0].cpu().numpy()  # (778, 3)
                joints = mano_out.joints[0].cpu().numpy()  # (21, 3)
                
                # 左手X轴翻转
                verts[:, 0] *= -1
                joints[:, 0] *= -1
                
                # 应用全局旋转和平移：R @ (V - J0) + T
                wrist = joints[0]  # 手腕位置
                global_orient_np = global_orient[0].cpu().numpy()  # (3, 3)
                transl_np = transl[0].cpu().numpy()  # (3,)
                
                # 计算相机坐标系下的顶点
                verts_cam = (global_orient_np @ (verts - wrist).T).T + transl_np
                
                verts_left_list[t] = verts_cam
                mask_left[t] = 1
            
            # 处理右手
            if t in recon_results.get('right', {}):
                res = recon_results['right'][t]
                
                # 提取参数
                beta = torch.from_numpy(res['beta']).unsqueeze(0).to(device)
                hand_pose = torch.from_numpy(res['hand_pose']).unsqueeze(0).to(device)
                global_orient = torch.from_numpy(res['global_orient']).unsqueeze(0).to(device)
                transl = torch.from_numpy(res['transl']).unsqueeze(0).to(device)
                
                # MANO前向传播
                identity_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).to(device)
                mano_out = mano(betas=beta, hand_pose=hand_pose, global_orient=identity_rot)
                verts = mano_out.vertices[0].cpu().numpy()
                joints = mano_out.joints[0].cpu().numpy()
                
                # 应用全局旋转和平移
                wrist = joints[0]
                global_orient_np = global_orient[0].cpu().numpy()
                transl_np = transl[0].cpu().numpy()
                
                verts_cam = (global_orient_np @ (verts - wrist).T).T + transl_np
                
                verts_right_list[t] = verts_cam
                mask_right[t] = 1

        # 3. 准备相机参数
        # 世界坐标系 = 相机坐标系
        # R = Identity, T = 0
        R_w2c = np.repeat(np.eye(3, dtype=np.float32)[None, ...], T, axis=0)  # (T, 3, 3)
        t_w2c = np.zeros((T, 3, 1), dtype=np.float32)  # (T, 3, 1)
        
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        
        # 4. 初始化渲染器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("警告: 未检测到 CUDA，渲染可能较慢或不支持。")
            
        try:
            renderer = Renderer(W, H, (fx, fy), device)
        except Exception as e:
            print(f"初始化渲染器失败: {e}")
            return

        # 5. 执行渲染
        # 构造基类需要的参数元组
        hand_traj_wordspace = (verts_left_list, verts_right_list)
        hand_mask = (mask_left, mask_right)
        extrinsics = (R_w2c, t_w2c)
        
        print("正在渲染...")
        try:
            # 仅使用 'cam' 模式渲染
            rendered_frames = self._render_hand_trajectory(
                video_frames=images,  # 此处传入原始图像列表
                hand_traj_wordspace=hand_traj_wordspace,
                hand_mask=hand_mask,
                extrinsics=extrinsics,
                renderer=renderer,
                mode='cam'
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"渲染过程中出错: {e}")
            return
        
        # 6. 保存视频
        print(f"正在保存视频到 {output_path} ...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用 OpenCV 保存视频，确保色彩空间正确 (RGB -> BGR)
        # 使用 H.264 编码器以获得更好的兼容性（VSCode、浏览器等都支持）
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 或者使用 'H264', 'X264'
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        if not out.isOpened():
            print("警告: 无法使用 H.264 编码器，尝试使用 mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        for frame in rendered_frames:
            # BaseHandVisualizer 返回的是 RGB 格式 (在 _render_hand_trajectory 结尾处转换过)
            # OpenCV VideoWriter 需要 BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        print("✅ 视频生成完成！")
