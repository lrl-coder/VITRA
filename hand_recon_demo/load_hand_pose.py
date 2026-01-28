"""
加载和使用手部位姿数据的示例脚本

演示如何：
1. 加载保存的手部位姿数据
2. 提取手腕的6D位姿（位置+旋转）
3. 提取手指的15个关节角度
4. 将旋转矩阵转换为其他表示形式（欧拉角、轴角）
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_to_euler(rot_matrix, seq='xyz', degrees=True):
    """
    将旋转矩阵转换为欧拉角
    
    参数:
        rot_matrix: (3, 3) 旋转矩阵
        seq: 欧拉角序列，如 'xyz', 'zyx' 等
        degrees: 是否以度为单位（False则为弧度）
    
    返回:
        euler_angles: (3,) 欧拉角 [roll, pitch, yaw]
    """
    r = Rotation.from_matrix(rot_matrix)
    return r.as_euler(seq, degrees=degrees)


def rotation_matrix_to_axis_angle(rot_matrix):
    """
    将旋转矩阵转换为轴角表示
    
    参数:
        rot_matrix: (3, 3) 旋转矩阵
    
    返回:
        axis_angle: (3,) 轴角向量，方向为旋转轴，长度为旋转角度（弧度）
    """
    r = Rotation.from_matrix(rot_matrix)
    return r.as_rotvec()


def load_and_analyze_hand_pose(pose_file_path):
    """
    加载并分析手部位姿数据
    
    参数:
        pose_file_path: 手部位姿数据文件路径（.npy）
    """
    print("="*60)
    print("加载手部位姿数据")
    print("="*60)
    
    # 1. 加载数据
    data = np.load(pose_file_path, allow_pickle=True).item()
    
    print(f"\n数据结构:")
    print(f"  - 左手帧数: {len(data['left'])}")
    print(f"  - 右手帧数: {len(data['right'])}")
    
    print(f"\n说明: {data['description']['note']}")
    print(f"顶点计算: {data['description']['usage']}")
    
    # 2. 提取第一帧的左手数据（如果存在）
    if len(data['left']) > 0:
        first_frame_idx = list(data['left'].keys())[0]
        left_hand = data['left'][first_frame_idx]
        
        print(f"\n左手第 {first_frame_idx} 帧数据:")
        print("-"*60)
        
        # 手腕位置（6D姿态的平移部分）
        wrist_pos = left_hand['wrist_position']
        print(f"\n手腕位置 (transl, 3D position):")
        print(f"  形状: {wrist_pos.shape}")
        print(f"  值: {wrist_pos}")
        print(f"  含义: [x, y, z] 在相机坐标系中的位置（米），已修正")
        
        # 手腕旋转（6D姿态的旋转部分）
        wrist_rot = left_hand['wrist_rotation']
        print(f"\n手腕旋转 (global_orient, 3D rotation):")
        print(f"  形状: {wrist_rot.shape}")
        print(f"  旋转矩阵:\n{wrist_rot}")
        
        # 转换为欧拉角
        euler = rotation_matrix_to_euler(wrist_rot)
        print(f"\n  欧拉角 (XYZ, 度): {euler}")
        
        # 转换为轴角
        axis_angle = rotation_matrix_to_axis_angle(wrist_rot)
        print(f"  轴角 (弧度): {axis_angle}")
        
        # 手指关节旋转（15个关节）
        finger_rot = left_hand['finger_rotations']
        print(f"\n手指关节旋转 (hand_pose):")
        print(f"  形状: {finger_rot.shape}")
        print(f"  说明: 15个手指关节，每个关节一个3x3旋转矩阵")
        
        # MANO关节顺序：
        # 0-2: 拇指（Thumb MCP, PIP, DIP）
        # 3-5: 食指（Index MCP, PIP, DIP）
        # 6-8: 中指（Middle MCP, PIP, DIP）
        # 9-11: 无名指（Ring MCP, PIP, DIP）
        # 12-14: 小指（Pinky MCP, PIP, DIP）
        
        joint_names = [
            "拇指-掌指关节(MCP)", "拇指-近端指间关节(PIP)", "拇指-远端指间关节(DIP)",
            "食指-MCP", "食指-PIP", "食指-DIP",
            "中指-MCP", "中指-PIP", "中指-DIP",
            "无名指-MCP", "无名指-PIP", "无名指-DIP",
            "小指-MCP", "小指-PIP", "小指-DIP"
        ]
        
        print(f"\n各关节的欧拉角（度）:")
        for i, (name, rot) in enumerate(zip(joint_names, finger_rot)):
            euler = rotation_matrix_to_euler(rot)
            print(f"  {i:2d}. {name:20s}: [{euler[0]:6.1f}, {euler[1]:6.1f}, {euler[2]:6.1f}]")
        
        # 形状参数
        shape_params = left_hand['shape_params']
        print(f"\nMANO形状参数 (beta):")
        print(f"  形状: {shape_params.shape}")
        print(f"  值: {shape_params}")
        print(f"  说明: MANO模型的10个形状参数，控制手的大小和形状")
        
        # 示例：如何从这些参数重建顶点和关节
        print(f"\n" + "="*60)
        print("如何从这些参数重建3D手部网格:")
        print("="*60)
        print("""
from libs.models.mano_wrapper import MANO
import torch

# 加载MANO模型
mano = MANO(model_path='./weights/mano').cuda()

# 准备参数
beta = torch.from_numpy(shape_params).unsqueeze(0).cuda()
hand_pose = torch.from_numpy(finger_rot).unsqueeze(0).cuda()
global_orient = torch.from_numpy(wrist_rot).unsqueeze(0).unsqueeze(0).cuda()

# 使用单位旋转生成局部顶点
identity_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).cuda()
mano_out = mano(betas=beta, hand_pose=hand_pose, global_orient=identity_rot)
verts_local = mano_out.vertices[0].cpu().numpy()  # (778, 3)
joints_local = mano_out.joints[0].cpu().numpy()   # (21, 3)

# 左手需要X轴翻转
verts_local[:, 0] *= -1
joints_local[:, 0] *= -1

# 应用全局旋转和平移
wrist = joints_local[0]
verts_cam = (wrist_rot @ (verts_local - wrist).T).T + wrist_pos
joints_cam = (wrist_rot @ (joints_local - wrist).T).T + wrist_pos

# 现在 verts_cam 就是相机坐标系下的顶点 (778, 3)
# joints_cam 就是相机坐标系下的21个关节 (21, 3)
        """)
    
    # 3. 提取第一帧的右手数据（如果存在）
    if len(data['right']) > 0:
        first_frame_idx = list(data['right'].keys())[0]
        right_hand = data['right'][first_frame_idx]
        
        print(f"\n\n右手第 {first_frame_idx} 帧数据:")
        print("-"*60)
        print(f"手腕位置: {right_hand['wrist_position']}")
        print(f"手腕旋转形状: {right_hand['wrist_rotation'].shape}")
        print(f"手指关节形状: {right_hand['finger_rotations'].shape}")
    
    print("\n" + "="*60)
    print("数据加载完成！")
    print("="*60)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python load_hand_pose.py <pose_file.npy>")
        print("\n示例:")
        print("  python load_hand_pose.py ./output/hand_pose.npy")
        return
    
    pose_file = sys.argv[1]
    load_and_analyze_hand_pose(pose_file)


if __name__ == "__main__":
    main()
