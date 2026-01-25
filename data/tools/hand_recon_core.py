import numpy as np
import torch
import copy

from .utils_moge import MogePipeline
from .utils_hawor import HaworPipeline
from libs.models.mano_wrapper import MANO

class Config:
    """
    配置类，用于管理模型路径。
    默认路径可以被传入的命令行参数（args）覆盖。
    """
    def __init__(self, args=None):
        # --- 模型路径（可通过 CLI 参数覆盖） ---
        # HaWoR 手部姿态估计算法模型权重路径
        self.HAWOR_MODEL_PATH = getattr(args, 'hawor_model_path', './weights/hawor/checkpoints/hawor.ckpt')
        # 手部检测器模型路径
        self.DETECTOR_PATH = getattr(args, 'detector_path', './weights/hawor/external/detector.pt')
        # MoGe 深度/相机参数预测模型路径
        self.MOGE_MODEL_PATH = getattr(args, 'moge_model_path', 'Ruicheng/moge-2-vitl')
        # MANO 模型（手部参数化模型）基础数据路径
        self.MANO_PATH = getattr(args, 'mano_path', './weights/mano')


class HandReconstructor:
    """
    3D 手部重建核心流水线，结合了：
    1. 相机估计 (MoGe)：用于确定视场角 (FoV) 和焦距。
    2. 姿态与运动估计 (HaWoR)：结合 MANO 模型预测手部的形状、旋转和位移。
    """
    def __init__(
            self, 
            config: Config,
            device: torch.device = torch.device("cuda")
    ):
        """
        初始化重建流水线的各个组件。

        参数:
            config (Config): 包含模型路径的配置对象。
            device (torch.device): 指定运行设备（如 'cuda' 或 'cpu'）。
        """
        self.device = device
        
        # 初始化 HaWoR 流水线（负责手部关键点和姿态预测）
        self.hawor_pipeline = HaworPipeline(
            model_path=config.HAWOR_MODEL_PATH, detector_path=config.DETECTOR_PATH, device=device
        )
        # 初始化 MoGe 流水线（负责从单目图像估计相机内参 FoV）
        self.moge_pipeline = MogePipeline(
            model_name=config.MOGE_MODEL_PATH, device=device
        )
        # 初始化 MANO 手部层，用于根据姿态和形状参数生成手部网格
        self.mano = MANO(model_path=config.MANO_PATH).to(device)

    def recon(self, images: list, thresh: float = 0.5) -> dict:
        """
        执行完整的 3D 手部重建过程。

        主要步骤:
        1. 使用 MoGe 估计相机的 FoV（取所有帧的中位数以保证稳定性）。
        2. 根据 FoV 计算相机焦距。
        3. 使用 HaWoR 估计每一帧手部的位姿 (Pose)、形状 (Shape/Betas) 和位移 (Translation)。
        4. 全局位移修正：将 MANO 模型输出的手腕关节位置与 HaWoR 的位移预测对齐，重新计算全局位移。

        参数:
            images (list): 输入的图像序列（numpy 数组格式）。
            thresh (float): 手部检测置信度阈值。

        返回:
            dict: 包含左/右手重建结果（位态、网格、修正后的位移等）和 FoV 的字典。
        """

        N = len(images)
        if N == 0:
            return {'left': {}, 'right': {}, 'fov_x': None}
        
        H, W = images[0].shape[:2]

        # --- 1. 相机视场角 (FoV) 估计 (使用 MoGe) ---
        all_fov_x = []
        for i in range(N):
            img = images[i]
            # 预测单帧的水平视场角
            fov_x = self.moge_pipeline.infer(img)
            all_fov_x.append(fov_x)
        
        # 为了抵消帧间波动，取所有帧 FoV 的中位数
        fov_x = np.median(np.array(all_fov_x))
        # 根据 FoV 和图像宽度计算焦距 (Focal Length)
        img_focal = 0.5 * W / np.tan(0.5 * fov_x * np.pi / 180)

        # --- 2. 手部姿态与初始位移估计 (使用 HaWoR) ---
        # HaWoR 会返回包含姿态参数、形状参数和相对相机平面的位移信息
        recon_results = self.hawor_pipeline.recon(images, img_focal, thresh=thresh, single_image=(N==1))

        recon_results_new_transl = {'left': {}, 'right': {}, 'fov_x': fov_x}
        
        # --- 3. 重新计算全局位移 (MANO 关节对齐) ---
        # HaWoR 预测的位移通常是相对于检测框或特定参考点的，
        # 这里通过 MANO 模型的正向传播获取标准坐标系下的手腕位置，从而修正全局平移。
        for img_idx in range(N):
            for hand_type in ['left', 'right']:
                # 检查该帧是否存在对应的手部检测结果
                if hand_type == 'left':
                    if not img_idx in recon_results['left']:
                        continue
                    result = recon_results['left'][img_idx]
                else:
                    if not img_idx in recon_results['right']:
                        continue
                    result = recon_results['right'][img_idx]

                # 将预测结果（numpy）转换为 Tensor 以便输入 MANO 模型
                betas = torch.from_numpy(result['beta']).unsqueeze(0).to(self.device)
                hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).to(self.device)
                transl = torch.from_numpy(result['transl']).unsqueeze(0).to(self.device)  

                # 执行 MANO 正向传播：基于 betas (形状) 和 hand_pose (旋转) 生成手部 3D nodes
                model_output = self.mano(betas = betas, hand_pose = hand_pose)    
                verts_m = model_output.vertices[0]
                joints_m = model_output.joints[0]

                # 如果是左手，需要翻转 x 轴（因为 MANO 默认通常是右手模型，通过镜像得到左手）
                if hand_type == 'left':
                    verts_m[:,0] = -1*verts_m[:,0]
                    joints_m[:,0] = -1*joints_m[:,0]
                
                # 获取 MANO 坐标系下的手腕 (Wrist) 关节位置（通常索引为 0）
                wrist = joints_m[0]

                # 计算修正后的新位移：将 MANO 局部坐标系对手腕的偏移补偿到预测的位移中
                transl_new = wrist + transl

                # 深拷贝原始结果并更新位移信息
                result_new_transl = copy.deepcopy(result)
                result_new_transl['transl'] = transl_new[0].cpu().numpy()
                recon_results_new_transl[hand_type][img_idx] = result_new_transl
        
        return recon_results_new_transl