import os
import cv2
import numpy as np
import torch
import matplotlib as mpl

from .video_utils import (
    read_video_frames,
    resize_frames_to_long_side,
    save_to_video,
    add_overlay_text
)
from typing import Optional, List, Tuple
from libs.models.mano_wrapper import MANO
from .render_utils import Renderer

class Config:
    """
    配置类：管理文件路径、参数和可视化设置
    路径使用默认值初始化，但可以通过命令行参数覆盖
    """
    def __init__(self, args=None):
        # --- 路径配置 (可通过命令行参数覆盖) ---
        self.VIDEO_ROOT = getattr(args, 'video_root', 'data/examples/videos')  # 视频文件根目录
        self.LABEL_ROOT = getattr(args, 'label_root', 'data/examples/annotations')  # 标注文件根目录
        self.SAVE_PATH = getattr(args, 'save_path', 'data/examples/visualize')  # 可视化结果保存目录
        self.MANO_MODEL_PATH = getattr(args, 'mano_model_path', './weights/mano')  # MANO 手部模型路径

        # --- 固定参数 ---
        self.RENDER_SIZE_LONG_SIDE = 480  # 渲染时图像长边大小（像素）
        self.FPS = 15  # 输出视频帧率

        # --- 颜色和颜色映射配置 ---
        self.LEFT_CMAP = "inferno"  # 左手轨迹颜色映射（渐变模式）
        self.RIGHT_CMAP = "inferno"  # 右手轨迹颜色映射（渐变模式）

        # 手部基础颜色（RGB，归一化到 0-1）
        self.LEFT_COLOR = np.array([0.6594, 0.6259, 0.7451])  # 左手浅紫色
        self.RIGHT_COLOR = np.array([0.4078, 0.4980, 0.7451])  # 右手深蓝色



class HandVisualizer:
    """
    手部可视化器主类
    功能：加载数据、配置渲染器，并可视化手部片段（包括网格和轨迹）
    """
    def __init__(self, config: Config, render_gradual_traj: bool = False):
        """
        初始化可视化器
        
        参数:
            config: 配置对象，包含路径和参数
            render_gradual_traj: 是否启用渐进式轨迹渲染（完整模式）
        """
        self.config = config
        self.render_gradual_traj = render_gradual_traj
        
        # 渲染模式列表
        self.all_modes = ['cam', 'first']  # 默认：相机模式 + 首帧模式
        if self.render_gradual_traj:
            self.all_modes = ['cam', 'full', 'first']  # 添加完整轨迹模式
        
        # 初始化 MANO 模型（默认使用右手模型）
        self.mano = MANO(model_path=self.config.MANO_MODEL_PATH).cuda()
        faces_right = torch.from_numpy(self.mano.faces).float().cuda()
        
        # MANO 面片定义为右手，左手需要翻转顶点顺序（镜像）
        self.faces_left = faces_right[:, [0, 2, 1]]  # 左手：翻转顶点顺序实现镜像
        self.faces_right = faces_right  # 右手：保持原始顺序

    def _render_hand_trajectory(self, video_frames, hand_traj_wordspace, hand_mask, extrinsics, renderer: Renderer, mode: str):
        """
        渲染手部网格轨迹
        
        根据不同模式渲染单帧手部网格或多帧手部轨迹：
        - 'cam' 模式：每帧仅渲染当前帧的手部网格
        - 'first' 模式：将完整轨迹渲染到第一帧上
        - 'full' 模式：渐进式轨迹渲染（从当前帧到结束帧）
        
        参数:
            video_frames: 视频帧列表 (T, H, W, 3)
            hand_traj_wordspace: 手部轨迹（世界坐标系）元组 (左手顶点, 右手顶点)
            hand_mask: 手部掩码元组 (左手掩码, 右手掩码)，标记哪些帧有有效手部
            extrinsics: 相机外参元组 (旋转矩阵 R_w2c, 平移向量 t_w2c)
            renderer: 渲染器实例
            mode: 渲染模式 ('cam', 'first', 'full')
            
        返回:
            all_save_frames: 渲染后的帧列表
        """
        # 解包输入数据
        verts_left_worldspace, verts_right_worldspace = hand_traj_wordspace
        left_hand_mask, right_hand_mask = hand_mask
        R_w2c, t_w2c = extrinsics

        num_total_frames = len(video_frames)
        all_save_frames = []

        # === 根据渲染模式确定参数 ===
        if mode == 'cam':
            # 相机模式：仅渲染当前帧的手部网格
            num_loop_frames = num_total_frames  # 需要循环处理所有帧
            # 所有帧使用相同的单一颜色
            left_colors = self.config.LEFT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
            right_colors = self.config.RIGHT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
        elif mode == 'first':
            # 首帧模式：将完整轨迹渲染到第一帧上
            num_loop_frames = 1  # 只处理第一帧
            left_colors = self.config.LEFT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
            right_colors = self.config.RIGHT_COLOR[np.newaxis, :].repeat(num_total_frames, axis=0)
        elif mode == 'full':
            # 完整模式：渐进式轨迹渲染
            num_loop_frames = num_total_frames
            # 为轨迹生成颜色序列（渐变色）
            left_colors, right_colors = generate_hand_colors(num_total_frames, self.config.LEFT_CMAP, self.config.RIGHT_CMAP)
        else:
            raise ValueError(f'Unknown rendering mode: {mode}')


        # === 主渲染循环 ===
        for current_frame_idx in range(num_loop_frames):

            if not mode == 'first':
                print(f'Processing frame {current_frame_idx + 1}/{num_loop_frames}', end='\r')
                # 从基础视频帧开始（复制并归一化到 0-1）
                curr_img_overlay = video_frames[current_frame_idx].copy().astype(np.float32) / 255.0

            # === 坐标系转换：世界坐标系 → 相机坐标系 ===
            # 获取当前帧的相机外参（旋转和平移）
            R_w2c_cur = R_w2c[current_frame_idx]  # (3, 3) 旋转矩阵
            t_w2c_cur = t_w2c[current_frame_idx]  # (3, 1) 平移向量

            # 将所有帧的手部顶点从世界坐标系转换到当前相机坐标系
            # 公式: V_cam = R_w2c @ V_world + t_w2c
            # 左手顶点转换: (T, 778, 3) -> (3, 3) @ (T, 3, 778) + (3, 1) -> (T, 3, 778) -> (T, 778, 3)
            verts_left_camspace = (
                R_w2c_cur @ verts_left_worldspace.transpose(0, 2, 1) + t_w2c_cur
            ).transpose(0, 2, 1)
            # 右手顶点转换（同样的操作）
            verts_right_camspace = (
                R_w2c_cur @ verts_right_worldspace.transpose(0, 2, 1) + t_w2c_cur
            ).transpose(0, 2, 1)

            # === 确定当前帧需要渲染的轨迹段 ===
            if mode == 'cam':
                # 相机模式：仅渲染当前帧的网格（索引范围：current -> current+1）
                start_traj_idx = current_frame_idx
                end_traj_idx = current_frame_idx + 1
                transparency = [1.0]  # 完全不透明
            elif mode == 'first':
                # 首帧模式：在第0帧上渲染完整轨迹（索引范围：0 -> T）
                start_traj_idx = 0
                end_traj_idx = num_total_frames
                transparency = [1.0] * (end_traj_idx - start_traj_idx)  # 所有轨迹点完全不透明
                # 'first' 模式循环只运行一次
                if current_frame_idx > 0: continue
            elif mode == 'full':
                # 完整模式：渐进式轨迹（索引范围：current -> T）
                start_traj_idx = current_frame_idx
                end_traj_idx = num_total_frames
                # 渐进式透明度：较早的轨迹点更透明（0.4 -> 0.7）
                transparency = np.linspace(0.4, 0.7, end_traj_idx - start_traj_idx)
            else:
                raise ValueError(f'Unknown rendering mode: {mode}')

            # === 遍历轨迹段的每个点 ===
            for traj_idx, kk in enumerate(range(start_traj_idx, end_traj_idx)):

                if mode == 'first':
                    print(f'Processing frame {traj_idx + 1}/{num_total_frames}', end='\r')
                    curr_img_overlay = video_frames[current_frame_idx].copy().astype(np.float32)/255

                # 获取轨迹点 'kk' 的手部数据
                left_mask_k = left_hand_mask[kk]  # 左手掩码（0=无手，1=有手）
                right_mask_k = right_hand_mask[kk]  # 右手掩码
                transp_k = transparency[traj_idx] if len(transparency) > traj_idx else 1.0  # 当前点的透明度

                # 初始化手部数据列表
                left_verts_list, left_color_list, left_face_list = ([], [], [])
                right_verts_list, right_color_list, right_face_list = ([], [], [])

                # === 准备左手渲染数据 ===
                if left_mask_k != 0:  # 如果该帧有左手
                    left_verts_list = [torch.from_numpy(verts_left_camspace[kk]).float().cuda()]  # (778, 3) 顶点
                    # 将颜色复制到所有 778 个顶点: (3,) -> (1, 3) -> (778, 3)
                    left_color_list = [torch.from_numpy(left_colors[kk]).float().unsqueeze(0).repeat(778, 1).cuda()]
                    left_face_list = [self.faces_left]  # 左手面片（镜像版本）

                # === 准备右手渲染数据 ===
                if right_mask_k != 0:  # 如果该帧有右手
                    right_verts_list = [torch.from_numpy(verts_right_camspace[kk]).float().cuda()]  # (778, 3) 顶点
                    # 同样将颜色复制到所有顶点
                    right_color_list = [torch.from_numpy(right_colors[kk]).float().unsqueeze(0).repeat(778, 1).cuda()]
                    right_face_list = [self.faces_right]  # 右手面片

                # 合并左右手的渲染数据
                verts_list  = left_verts_list + right_verts_list  # 顶点列表
                faces_list  = left_face_list + right_face_list     # 面片列表
                colors_list = left_color_list + right_color_list   # 颜色列表

                # === 渲染手部网格 ===
                if verts_list:  # 如果有手部数据需要渲染
                    # 调用渲染器生成网格图像
                    rend, mask = renderer.render(verts_list, faces_list, colors_list)
                    rend = rend[..., ::-1]  # 颜色空间转换：RGB -> BGR

                    color_mesh = rend.astype(np.float32) / 255.0  # 网格图像（归一化到 0-1）
                    valid_mask = mask[..., None].astype(np.float32)  # 有效像素掩码 (H, W, 1)

                    # === Alpha 混合：将网格叠加到视频帧上 ===
                    # 公式: 新图像 = 原图像 × (1-掩码) + 网格 × 掩码 × 透明度 + 原图像 × 掩码 × (1-透明度)
                    # 这样可以实现平滑的半透明叠加效果
                    curr_img_overlay = (
                        curr_img_overlay[:, :, :3] * (1 - valid_mask) +  # 非手部区域：保持原图像
                        color_mesh[:, :, :3] * valid_mask * transp_k +   # 手部区域：网格部分
                        curr_img_overlay[:, :, :3] * valid_mask * (1 - transp_k)  # 手部区域：原图像部分
                    )
                if mode == 'first':
                    # 首帧模式：每个轨迹点处理后立即保存帧
                    final_frame = (curr_img_overlay * 255).astype(np.uint8)  # 反归一化到 0-255
                    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                    all_save_frames.append(final_frame)
            
            if mode == 'cam' or mode == 'full':
                # 相机模式/完整模式：每个当前帧处理完后保存
                final_frame = (curr_img_overlay * 255).astype(np.uint8)  # 反归一化到 0-255
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                all_save_frames.append(final_frame)

        print(f'Finished rendering with mode: {mode}')
        return all_save_frames  # 返回所有渲染后的帧

    def process_episode(self, episode_name: str):
            """加载数据并协调单个片段的可视化过程。"""
            print(f'\n正在处理片段: {episode_name}')

            # 1. 加载路径并检查是否存在
            # 格式解析：dataset_name_ep_name_video_name
            dataset_name = episode_name.split('_')[0]
            ep_name = episode_name.split('_')[-2] + '_' + episode_name.split('_')[-1]
            video_name = episode_name.replace(f'{dataset_name}_', '').replace(f'_{ep_name}', '')
            video_path = os.path.join(self.config.VIDEO_ROOT, f'{video_name}.mp4')
            label_path = os.path.join(self.config.LABEL_ROOT, episode_name + '.npy')

            if not os.path.exists(label_path):
                print(f'片段文件 {label_path} 不存在，跳过...')
                return

            # 2. 加载片段数据
            cap = cv2.VideoCapture(video_path)
            episode_info = np.load(label_path, allow_pickle=True).item()

            # 获取帧范围、相机内外参、文本说明和手部属性
            start_frame, end_frame = get_frame_interval(episode_info)
            R_w2c, t_w2c, normalized_intrinsics = get_camera_info(episode_info)
            caption_left, caption_right, hand_type = get_caption_info(episode_info)
            
            # 将 MANO 参数转换为世界坐标系下的顶点
            (verts_left_worldspace, left_hand_mask), (verts_right_worldspace, right_hand_mask) = \
                get_hand_labels(episode_info, self.mano)

            # 3. 读取并缩放视频帧
            video_frames = read_video_frames(cap, start_frame=start_frame, end_frame=end_frame, interval=1)
            resize_video_frames = resize_frames_to_long_side(video_frames, self.config.RENDER_SIZE_LONG_SIDE)
            H, W, _ = resize_video_frames[0].shape

            # 4. 初始化渲染器
            # 根据新的图像尺寸 (W, H) 反归一化内参
            intrinsics_denorm = normalized_intrinsics.copy()
            intrinsics_denorm[0] *= W
            intrinsics_denorm[1] *= H
            fx_exo = intrinsics_denorm[0, 0]
            fy_exo = intrinsics_denorm[1, 1]

            renderer = Renderer(W, H, (fx_exo, fy_exo), 'cuda')

            # 5. 渲染各模式下的手部
            all_rendered_frames = []
            hand_traj_wordspace = (verts_left_worldspace, verts_right_worldspace)
            hand_mask = (left_hand_mask, right_hand_mask)
            extrinsics = (R_w2c, t_w2c)

            for mode in self.all_modes:
                save_frames = self._render_hand_trajectory(
                    resize_video_frames,
                    hand_traj_wordspace,
                    hand_mask,
                    extrinsics,
                    renderer,
                    mode=mode
                )
                all_rendered_frames.append(save_frames)

            # 6. 拼接帧并添加文字说明
            final_save_frames = []
            num_frames = len(all_rendered_frames[0])

            # 选择主要手部的文字，并提取反向区间用于文字查找
            caption_primary = caption_right if hand_type == 'right' else caption_left
            caption_opposite = caption_left if hand_type == 'right' else caption_right
            opposite_intervals = [interval for _, interval in caption_opposite]

            for frame_idx in range(num_frames):
                # 横向拼接不同模式渲染出的图像
                curr_img_overlay = np.concatenate(
                    [all_rendered_frames[mode_idx][frame_idx] for mode_idx in range(len(self.all_modes))],
                    axis=1
                )

                # 获取主手部的文字说明（假设主手部说明只有一个区间）
                overlay_text_primary = caption_primary[0][0]

                # 根据当前帧索引查找辅助手部的文字说明
                opposite_idx = find_caption_index(frame_idx, opposite_intervals)
                overlay_text_opposite = caption_opposite[opposite_idx][0] if opposite_idx is not None else 'None.'

                # 格式化并添加完整的叠加文字
                overlay_text_full = generate_overlay_text(
                    overlay_text_primary, 
                    overlay_text_opposite, 
                    hand_type
                )
                add_overlay_text(curr_img_overlay, overlay_text_full)

                final_save_frames.append(curr_img_overlay)

            # 7. 保存最终视频
            os.makedirs(self.config.SAVE_PATH, exist_ok=True)
            save_to_video(final_save_frames, f'{self.config.SAVE_PATH}/{episode_name}.mp4', fps=self.config.FPS)
            print(f'\n成功保存片段至 {self.config.SAVE_PATH}/{episode_name}.mp4')

def find_caption_index(frame_index: int, intervals: list[tuple[int, int]]) -> Optional[int]:
    """根据帧索引在区间列表中查找对应的索引位置。"""
    for idx, (start, end) in enumerate(intervals):
        if start <= frame_index <= end:
            return idx
    return None

def generate_hand_colors(T: int, left_cmap: str, right_cmap: str) -> tuple[np.ndarray, np.ndarray]:
    """
    为左手和右手生成 T 帧的 RGB 颜色序列。
    基于指定的 colormap 返回形状为 (T, 3) 且归一化到 0-1 的颜色数组。
    """
    t_norm = np.linspace(0, 0.95, T)
    left_colors = mpl.colormaps.get_cmap(left_cmap)(t_norm)[:, :3]
    right_colors = mpl.colormaps.get_cmap(right_cmap)(t_norm)[:, :3]
    return left_colors, right_colors

def get_frame_interval(episode_info: dict) -> tuple[int, int]:
    """从片段信息中提取起始（闭）和结束（开）帧索引。"""
    video_decode_frames = episode_info['video_decode_frame']
    start_frame = video_decode_frames[0]
    end_frame = video_decode_frames[-1] + 1
    return start_frame, end_frame

def normalize_camera_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """
    归一化相机内参。
    假设主点（principal point）位于图像中心（图像大小为 2*cx, 2*cy）。
    """
    # 深度拷贝以避免修改原始数组
    normalized_intrinsics = intrinsics.copy()
    normalized_intrinsics[0] /= normalized_intrinsics[0, 2] * 2
    normalized_intrinsics[1] /= normalized_intrinsics[1, 2] * 2
    return normalized_intrinsics

def get_camera_info(episode_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取并归一化相机内参和外参（世界到相机坐标系）。
    """
    extrinsics = episode_info['extrinsics']  # world2cam, 形状为 (T, 4, 4)
    R_w2c = extrinsics[:, :3, :3].copy()
    t_w2c = extrinsics[:, :3, 3:].copy()  # 形状为 (T, 3, 1)

    intrinsics = episode_info['intrinsics'].copy()
    normalized_intrinsics = normalize_camera_intrinsics(intrinsics)

    return R_w2c, t_w2c, normalized_intrinsics

def get_caption_info(episode_info: dict) -> tuple[list, list, str]:
    """
    提取并格式化左右手的文字说明信息。
    如果说明为空，则添加一个大的区间覆盖所有帧。
    """
    hand_type = episode_info['anno_type']

    caption_right = episode_info['text'].get('right', [])
    caption_left = episode_info['text'].get('left', [])

    # 确保文字说明不为空，以简化后续逻辑
    if not caption_right:
        caption_right = [['None.', (0, 10000)]] # 大区间覆盖所有帧
    if not caption_left:
        caption_left = [['None.', (0, 10000)]]

    return caption_left, caption_right, hand_type

def get_hand_labels(episode_info: dict, mano: MANO):
    """
    通过 MANO 模型处理手部标签（姿态、形状、平移、朝向）以获取世界空间下的手部顶点。
    """
    left_labels = episode_info['left']
    right_labels = episode_info['right']

    # --- 左手处理 ---
    left_hand_mask = left_labels['kept_frames']
    verts_left, _ = process_single_hand_labels(left_labels, left_hand_mask, mano, is_left=True)

    # --- 右手处理 ---
    right_hand_mask = right_labels['kept_frames']
    verts_right, _ = process_single_hand_labels(right_labels, right_hand_mask, mano)
    
    return (verts_left, left_hand_mask), (verts_right, right_hand_mask)

def process_single_hand_labels(hand_labels: dict, hand_mask: np.ndarray, mano: MANO, is_left: bool = False):
    """
    辅助函数：计算单只手（左手或右手）的 MANO 顶点。
    """
    T = len(hand_mask)
    
    wrist_worldspace = hand_labels['transl_worldspace'].reshape(-1, 1, 3)
    wrist_orientation = hand_labels['global_orient_worldspace']
    beta = hand_labels['beta']
    pose = hand_labels['hand_pose']

    # 为没有手部的帧（mask为0）设置恒等姿态
    identity = np.eye(3, dtype=pose.dtype)
    identity_block = np.broadcast_to(identity, (pose.shape[1], 3, 3))
    mask_indices = (hand_mask == 0)
    if np.any(mask_indices):
        pose[mask_indices] = identity_block

    beta_torch = torch.from_numpy(beta).float().cuda().unsqueeze(0).repeat(T, 1)
    pose_torch = torch.from_numpy(pose).float().cuda()
    
    # MANO 前向传播的全局旋转占位符（稍后手动应用）
    global_rot_placeholder = torch.eye(3).float().unsqueeze(0).unsqueeze(0).cuda().repeat(T, 1, 1, 1)
    # MANO 模型前向计算
    mano_out = mano(betas=beta_torch, hand_pose=pose_torch, global_orient=global_rot_placeholder)
    
    verts = mano_out.vertices.cpu().numpy()
    joints = mano_out.joints.cpu().numpy()

    # 应用 X 轴翻转（针对左手）
    if is_left:
        verts[:, :, 0] *= -1
        joints[:, :, 0] *= -1

    # 转换到世界空间坐标：R @ (V - J0) + T
    # (T, 778, 3) = (T, 3, 3) @ (T, 3, 778) + (T, 3, 1) -> (T, 3, 778) -> (T, 778, 3)
    verts_worldspace = (
        wrist_orientation @ 
        (verts - joints[:, 0][:, None]).transpose(0, 2, 1)
    ).transpose(0, 2, 1) + wrist_worldspace

    return verts_worldspace, joints[:, 0]

def generate_overlay_text(overlay_text: str, overlay_text_opposite: str, hand_type: str) -> str:
    """根据主手部类型格式化最终的文字说明字符串。"""
    if hand_type == 'right':
        return f'Left: {overlay_text_opposite} | Right: {overlay_text}'
    else:
        return f'Left: {overlay_text} | Right: {overlay_text_opposite}'