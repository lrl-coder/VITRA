"""
3D手部重建核心模块（已知相机内参版本）

本模块实现了使用已知相机内参进行3D手部重建的流水线。
与 hand_recon_core.py 的主要区别：
1. 不需要使用 MoGe 来估计相机参数
2. 直接使用用户提供的相机内参矩阵 K
3. 简化了处理流程，更适合已知相机参数的场景
"""

import numpy as np
import torch
import copy
from typing import Dict, List, Optional, Tuple


class HandReconstructorWithKnownCamera:
    """
    基于已知相机内参的3D手部重建流水线
    
    主要功能：
    1. 使用 HaWoR 进行手部姿态估计
    2. 使用 MANO 模型生成手部网格
    3. 基于已知相机内参进行3D重建
    """
    
    def __init__(
        self,
        hawor_model_path: str = './weights/hawor/checkpoints/hawor.ckpt',
        detector_path: str = './weights/hawor/external/detector.pt',
        mano_path: str = './weights/mano',
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        初始化手部重建器
        
        参数:
            hawor_model_path: HaWoR模型权重路径
            detector_path: 手部检测器路径
            mano_path: MANO模型路径
            device: 运行设备（cuda/cpu）
        """
        self.device = device
        
        # 导入依赖（延迟导入以避免不必要的加载）
        try:
            from data.tools.utils_hawor import HaworPipeline
            from libs.models.mano_wrapper import MANO
        except ImportError as e:
            raise ImportError(f"无法导入必要的模块: {e}")
        
        # 初始化 HaWoR 流水线（手部姿态估计）
        self.hawor_pipeline = HaworPipeline(
            model_path=hawor_model_path,
            detector_path=detector_path,
            device=device
        )
        
        # 初始化 MANO 手部模型
        self.mano = MANO(model_path=mano_path).to(device)
    
    def recon(
        self,
        images: List[np.ndarray],
        camera_intrinsics: np.ndarray,
        thresh: float = 0.5
    ) -> Dict:
        """
        执行手部重建
        
        参数:
            images: 图像序列，每个图像为 (H, W, 3) 的numpy数组
            camera_intrinsics: 相机内参矩阵 K，形状为 (3, 3)
                             格式: [[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]]
            thresh: 手部检测置信度阈值
        
        返回:
            dict: 包含左右手重建结果的字典
                {
                    'left': {frame_idx: {手部参数}},
                    'right': {frame_idx: {手部参数}},
                    'camera_intrinsics': K
                }
        """
        N = len(images)
        if N == 0:
            return {'left': {}, 'right': {}, 'camera_intrinsics': camera_intrinsics}
        
        H, W = images[0].shape[:2]
        
        # 从相机内参矩阵提取焦距
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        img_focal = (fx + fy) / 2.0  # 使用平均焦距
        
        print(f"使用相机内参: fx={fx:.2f}, fy={fy:.2f}, 平均焦距={img_focal:.2f}")
        
        # 使用 HaWoR 进行手部姿态估计
        recon_results = self.hawor_pipeline.recon(
            images,
            img_focal,
            thresh=thresh,
            single_image=(N == 1)
        )
        
        # 重新计算全局位移，对齐到 MANO 坐标系
        recon_results_aligned = {'left': {}, 'right': {}, 'camera_intrinsics': camera_intrinsics}
        
        for img_idx in range(N):
            for hand_type in ['left', 'right']:
                # 检查该帧是否检测到对应手部
                if img_idx not in recon_results[hand_type]:
                    continue
                
                result = recon_results[hand_type][img_idx]
                
                # 转换为 Tensor
                betas = torch.from_numpy(result['beta']).unsqueeze(0).to(self.device)
                hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).to(self.device)
                transl = torch.from_numpy(result['transl']).unsqueeze(0).to(self.device)
                
                # MANO 正向传播：生成手部顶点和关节点
                model_output = self.mano(betas=betas, hand_pose=hand_pose)
                verts = model_output.vertices[0]
                joints = model_output.joints[0]
                
                # 左手需要镜像翻转
                if hand_type == 'left':
                    verts[:, 0] = -verts[:, 0]
                    joints[:, 0] = -joints[:, 0]
                
                # 获取手腕关节（索引0）作为参考点
                wrist = joints[0]
                
                # 计算对齐后的全局位移
                transl_aligned = wrist + transl
                
                # 深拷贝结果并更新位移
                result_aligned = copy.deepcopy(result)
                result_aligned['transl'] = transl_aligned[0].cpu().numpy()
                result_aligned['vertices'] = verts.cpu().numpy()
                result_aligned['joints'] = joints.cpu().numpy()
                
                recon_results_aligned[hand_type][img_idx] = result_aligned
        
        return recon_results_aligned
    
    def project_to_2d(
        self,
        points_3d: np.ndarray,
        camera_intrinsics: np.ndarray
    ) -> np.ndarray:
        """
        将3D点投影到2D图像平面
        
        参数:
            points_3d: 3D点坐标，形状 (N, 3)
            camera_intrinsics: 相机内参矩阵，形状 (3, 3)
        
        返回:
            points_2d: 2D投影点，形状 (N, 2)
        """
        # 投影公式: p_2d = K @ p_3d
        # 其中 p_3d 是齐次坐标 [x, y, z, 1]
        points_2d_homo = camera_intrinsics @ points_3d.T  # (3, N)
        points_2d = points_2d_homo[:2] / points_2d_homo[2]  # 归一化
        return points_2d.T  # (N, 2)
