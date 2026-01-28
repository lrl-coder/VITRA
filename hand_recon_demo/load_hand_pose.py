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
    
    # 2. 提取第一帧的左手数据（如果存在）
    if len(data['left']) > 0:
        first_frame_idx = list(data['left'].keys())[0]
        left_hand = data['left'][first_frame_idx]
        
        print(f"\n左手第 {first_frame_idx} 帧数据:")
        print("-"*60)
        
        # 手腕位置（6D姿态的平移部分）
        wrist_pos = left_hand['wrist_position']
        print(f"\n手腕位置 (3D position):")
        print(f"  形状: {wrist_pos.shape}")
        print(f"  值: {wrist_pos}")
        print(f"  含义: [x, y, z] 在相机坐标系中的位置（米）")
        
        # 手腕旋转（6D姿态的旋转部分）
        wrist_rot = left_hand['wrist_rotation']
        print(f"\n手腕旋转 (3D rotation):")
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
        print(f"\n手指关节旋转:")
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
        
        # 关节3D坐标
        joints_3d = left_hand['joints_3d']
        print(f"\n手部关节3D坐标:")
        print(f"  形状: {joints_3d.shape}")
        print(f"  说明: 21个关节的3D坐标（包括手腕）")
        print(f"  前5个关节坐标:")
        for i in range(min(5, len(joints_3d))):
            print(f"    关节 {i}: {joints_3d[i]}")
    
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
