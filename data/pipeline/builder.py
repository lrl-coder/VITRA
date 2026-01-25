"""
Main Builder Class for the VITRA Dataset Pipeline.
Orchestrates the stages to process videos into the final dataset format.
"""

import os
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import concurrent.futures

from .config import PipelineConfig, load_config
from .stages import (
    VideoProcessor, 
    HandReconstructionStage,
    ActionSegmentationStage, 
    LanguageAnnotationStage,
    VisualizationStage
)

# Try to import CameraPoseEstimationStage (optional, requires DROID-SLAM)
try:
    from .stages.camera_pose import CameraPoseEstimationStage
    HAS_CAMERA_POSE = True
except ImportError:
    HAS_CAMERA_POSE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DatasetBuilder:
    """
    Main controller for the dataset construction pipeline.
    
    Flow:
    1. Scan video directory.
    2. For each video:
       a. Extract Frames & Preprocess (VideoProcessor)
       b. Reconstruct 3D Hand/Camera Pose (HandReconstructionStage)
       c. Segment into Atomic Actions (ActionSegmentationStage)
       d. Generate Language Annotations (LanguageAnnotationStage)
       e. Save output .npy files (Metadata format)
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[PipelineConfig] = None):
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)
        self.logger = logging.getLogger("DatasetBuilder")
        
        # Initialize Stages
        self.logger.info("Initializing pipeline stages...")
        self.video_processor = VideoProcessor(self.config.video, self.logger)
        self.hand_recon = HandReconstructionStage(self.config.hand_recon, self.logger)
        self.camera_pose = None
        if HAS_CAMERA_POSE:
            self.camera_pose = CameraPoseEstimationStage(self.config, self.logger)
        self.segmenter = ActionSegmentationStage(self.config.segmentation, self.logger)
        self.annotator = LanguageAnnotationStage(self.config.annotation, self.logger)
        self.visualizer = VisualizationStage(self.config.output, self.logger)
        
        # Initialize all stages (load models etc)
        self._init_stages()

    def _init_stages(self):
        self.video_processor.initialize()
        self.hand_recon.initialize()
        if self.camera_pose:
            self.camera_pose.initialize()
        self.segmenter.initialize()
        self.annotator.initialize()
        self.visualizer.initialize()

    def pipleline_cleanup(self):
        self.video_processor.cleanup()
        self.hand_recon.cleanup()
        if self.camera_pose:
            self.camera_pose.cleanup()
        self.segmenter.cleanup()
        self.annotator.cleanup()
        self.visualizer.cleanup()

    def process_directory(self, input_dir: str, output_dir: str):
        """
        Process all videos in a directory OR a single video file.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find videos
        videos = []
        if input_path.is_file():
            # If input is a single file, just add it to the list
            self.logger.info(f"Processing single video file: {input_path}")
            # Check format manually since find_videos is skipped
            if input_path.suffix in self.config.video.supported_formats:
                videos = [input_path]
            else:
                 self.logger.warning(f"File {input_path} format {input_path.suffix} not in supported list: {self.config.video.supported_formats}")
                 # Try to process anyway? Or stop? Let's proceed but warn.
                 videos = [input_path]
                 
        elif input_path.is_dir():
            videos = self.video_processor.find_videos(input_path, self.config.video.supported_formats)
            self.logger.info(f"Found {len(videos)} videos in {input_dir}")
        else:
            self.logger.error(f"Input path {input_dir} is neither a file nor a directory")
            return
        
        # Process loop (Sequential for now, could be parallelized)
        # Note: GPU models (HandRecon) usually restrict parallel processing on single GPU
        for video_file in tqdm(videos, desc="Processing videos"):
            try:
                self.process_single_video(video_file, output_path)
            except Exception as e:
                self.logger.error(f"Failed to process {video_file.name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

    def process_single_video(self, video_path: Path, output_root: Path):
        """
        Run the full pipeline on a single video.
        
        Pipeline Steps:
        1. Video Processing - Extract frames
        2. Hand Reconstruction - 3D hand pose in camera-space (HaWoR)
        3. Camera Pose Estimation - Camera trajectory (DROID-SLAM)
        4. World-Space Transformation - Convert cam-space to world-space
        5. Action Segmentation - Detect atomic actions
        6. Language Annotation - Generate text descriptions
        7. Visualization - Render results
        8. Save Output - Export .npy files
        """
        self.logger.info(f"Starting processing for {video_path.name}")
        video_name = video_path.stem
        dataset_name = "custom"  # Default dataset name
        
        # --- Stage 1: Video Processing ---
        self.logger.info("Stage 1: Video Processing...")
        video_data = self.video_processor.process({
            "video_path": video_path
        })
        
        # --- Stage 2: Hand Reconstruction (Camera-Space) ---
        self.logger.info("Stage 2: Hand Reconstruction (Camera-Space)...")
        recon_data = self.hand_recon.process(video_data)
        
        # --- Stage 3: Camera Pose Estimation ---
        if self.camera_pose:
            self.logger.info("Stage 3: Camera Pose Estimation (SLAM)...")
            recon_data = self.camera_pose.process(recon_data)
        else:
            self.logger.info("Stage 3: Camera Pose Estimation (skipped - using static camera assumption)")
            # Add identity camera poses
            T = len(recon_data.get("frames", []))
            recon_data["camera_poses"] = self._get_identity_camera_poses(T)
        
        # --- Stage 4: World-Space Transformation ---
        self.logger.info("Stage 4: Transforming to World-Space...")
        recon_data = self._transform_to_world_space(recon_data)
        
        # --- Stage 5: Action Segmentation ---
        self.logger.info("Stage 5: Action Segmentation...")
        segmented_data = self.segmenter.process(recon_data)
        
        # --- Stage 6: Language Annotation ---
        self.logger.info("Stage 6: Language Annotation...")
        final_data = self.annotator.process(segmented_data)
        
        # --- Stage 7: Visualization (Optional) ---
        self.logger.info("Stage 7: Visualization...")
        final_data['output_dir'] = output_root
        final_data['video_name'] = video_name
        self.visualizer.process(final_data)
        
        # --- Stage 8: Save Output ---
        self.logger.info("Stage 8: Saving Results...")
        self._save_results(final_data, output_root, dataset_name, video_name)
        
        self.logger.info(f"Completed {video_path.name}")
    
    def _get_identity_camera_poses(self, T: int) -> Dict[str, np.ndarray]:
        """Return identity camera poses for T frames (static camera assumption)."""
        return {
            "R_c2w": np.tile(np.eye(3), (T, 1, 1)).astype(np.float64),
            "t_c2w": np.zeros((T, 3), dtype=np.float64),
            "R_w2c": np.tile(np.eye(3), (T, 1, 1)).astype(np.float64),
            "t_w2c": np.zeros((T, 3), dtype=np.float64),
            "extrinsics": np.tile(np.eye(4), (T, 1, 1)).astype(np.float64),
        }
    
    def _transform_to_world_space(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform hand reconstruction data from camera-space to world-space.
        
        This is a key step in the VITRA pipeline:
        - Hand poses from HaWoR are in camera-space
        - We need world-space for stable action representation
        - Uses camera poses from SLAM to perform the transformation
        """
        camera_poses = data.get("camera_poses", {})
        recon = data.get("reconstruction", {})
        
        if not recon or not camera_poses:
            return data
        
        R_c2w = camera_poses.get("R_c2w")  # (T, 3, 3)
        t_c2w = camera_poses.get("t_c2w")  # (T, 3)
        
        if R_c2w is None or t_c2w is None:
            return data
        
        # Transform each hand's data
        for hand_side in ['left', 'right']:
            hand_data = recon.get(hand_side, {})
            if not hand_data:
                continue
            
            # hand_data is {frame_id: {beta, global_orient, hand_pose, transl}}
            for frame_id, frame_data in hand_data.items():
                if frame_id >= len(R_c2w):
                    continue
                
                R = R_c2w[frame_id]  # (3, 3) camera-to-world rotation
                t = t_c2w[frame_id]  # (3,) camera-to-world translation
                
                # Transform global orientation (wrist rotation)
                if 'global_orient' in frame_data:
                    global_orient_cam = frame_data['global_orient']  # (3, 3)
                    if global_orient_cam.shape == (3, 3):
                        # World rotation = R_c2w @ R_cam
                        global_orient_world = R @ global_orient_cam
                        frame_data['global_orient_worldspace'] = global_orient_world
                        frame_data['global_orient_camspace'] = global_orient_cam.copy()
                
                # Transform translation (wrist position)
                if 'transl' in frame_data:
                    transl_cam = frame_data['transl']  # (3,)
                    # World position = R_c2w @ p_cam + t_c2w
                    transl_world = R @ transl_cam + t
                    frame_data['transl_worldspace'] = transl_world
                    frame_data['transl_camspace'] = transl_cam.copy()
        
        # Store the world-space transformed data
        data['reconstruction'] = recon
        
        return data

    def _save_results(self, data: Dict[str, Any], output_root: Path, dataset_name: str, video_name: str):
        """
        Save results in the format expected by VITRA human_dataset.py.
        
        Format:
        {dataset_name}_{video_name}_ep_{ep_idx}.npy
        """
        episodes = data.get("episodes", [])
        recon = data.get("reconstruction", {})
        camera_poses = data.get("camera_poses", {})
        
        # Get image dimensions for intrinsics calculation
        frames = data.get("frames", [])
        if frames:
            h, w = frames[0].shape[:2]
        else:
            h, w = data.get("height", 480), data.get("width", 640)
        
        # Try to get intrinsics, or compute from fov_x
        intrinsics = data.get("intrinsics")
        if intrinsics is None and recon:
            # Compute intrinsics from fov_x
            fov_x = recon.get("fov_x")
            if fov_x is not None:
                # fov_x is in degrees, compute focal length
                # f = w / (2 * tan(fov_x / 2))
                f = w / (2 * np.tan(np.deg2rad(fov_x) / 2))
                intrinsics = np.array([
                    [f, 0, w / 2],
                    [0, f, h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)
                self.logger.info(f"Computed intrinsics from fov_x={fov_x:.2f}°, focal_length={f:.2f}")
        
        # Ensure subdirectories
        save_dir = output_root / "episodic_annotations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert internal format to VITRA .npy format
        # See data/data.md section 4 for structure
        
        for i, ep in enumerate(episodes):
            ep_filename = f"{dataset_name}_{video_name}_ep_{i:06d}.npy"
            save_path = save_dir / ep_filename
            
            start = ep["start_frame"]
            end = ep["end_frame"]
            
            # Format episode with all required fields
            episode_dict = self._format_vitra_episode(
                ep, recon, intrinsics, camera_poses, start, end, (h, w)
            )
            
            np.save(save_path, episode_dict)
            self.logger.info(f"Saved episode {i}: {ep_filename}")
            
        # Update index file (optional, but needed for loading)
        # self._update_index_file(...)

    def _format_vitra_episode(self, ep_info, recon_full, intrinsics, camera_poses, start_idx, end_idx, image_size=None):
        """
        Helper to format a single episode slice into the VITRA dictionary format.
        
        The expected format is documented in data/data.md section 4.
        
        Args:
            ep_info: Episode information dict containing text, anno_type, etc.
            recon_full: Reconstruction results with 'left', 'right', 'fov_x' keys.
                        Each hand contains {frame_id: {beta, hand_pose, global_orient, transl,
                                            global_orient_worldspace, transl_worldspace, ...}}
            intrinsics: Camera intrinsic matrix (3x3)
            camera_poses: Dict with 'extrinsics', 'R_w2c', 't_w2c' arrays from SLAM
            start_idx: Start frame index
            end_idx: End frame index (exclusive)
            image_size: Tuple of (height, width) for default intrinsics calculation
            
        Returns:
            Dictionary in VITRA metadata format.
        """
        T = end_idx - start_idx
        frame_indices = list(range(start_idx, end_idx))
        
        # Build extrinsics array from camera poses (world-to-camera transforms)
        if camera_poses and "extrinsics" in camera_poses:
            # Extract extrinsics for this episode's frames
            all_extrinsics = camera_poses["extrinsics"]
            extrinsics = np.zeros((T, 4, 4), dtype=np.float64)
            for t_idx, frame_id in enumerate(frame_indices):
                if frame_id < len(all_extrinsics):
                    extrinsics[t_idx] = all_extrinsics[frame_id]
                else:
                    extrinsics[t_idx] = np.eye(4)
        else:
            # Fallback to identity (static camera assumption)
            extrinsics = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)
        
        # Build intrinsics - use provided or compute default from image size
        if intrinsics is not None:
            intrinsics_arr = np.array(intrinsics).astype(np.float64)
        else:
            # Create default intrinsics based on image size
            # Assuming a typical FOV of ~60 degrees
            if image_size is not None:
                h, w = image_size
            else:
                h, w = 480, 640  # Default resolution
            
            # Default focal length assuming ~60 degree horizontal FOV
            default_fov_deg = 60.0
            f = w / (2 * np.tan(np.deg2rad(default_fov_deg) / 2))
            intrinsics_arr = np.array([
                [f, 0, w / 2],
                [0, f, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
        
        def extract_hand_data(hand_side):
            """
            Extract and format hand data from reconstruction results.
            
            Returns dict with all required fields for one hand.
            """
            hand_recon = recon_full.get(hand_side, {})
            
            # Initialize arrays
            beta = np.zeros(10, dtype=np.float64)
            global_orient_camspace = np.zeros((T, 3, 3), dtype=np.float64)
            global_orient_worldspace = np.zeros((T, 3, 3), dtype=np.float64)
            hand_pose = np.zeros((T, 15, 3, 3), dtype=np.float64)
            transl_camspace = np.zeros((T, 3), dtype=np.float64)
            transl_worldspace = np.zeros((T, 3), dtype=np.float64)
            joints_camspace = np.zeros((T, 21, 3), dtype=np.float32)
            joints_worldspace = np.zeros((T, 21, 3), dtype=np.float64)
            wrist = np.zeros((T, 1, 3), dtype=np.float32)
            kept_frames = np.zeros(T, dtype=np.int64)
            
            # Default identity rotation for unused frames
            identity_rot = np.eye(3)
            for t in range(T):
                for j in range(15):
                    hand_pose[t, j] = identity_rot.copy()
                global_orient_camspace[t] = identity_rot.copy()
                global_orient_worldspace[t] = identity_rot.copy()
            
            # Fill in data from reconstruction results
            beta_set = False
            for t_idx, frame_id in enumerate(frame_indices):
                if frame_id in hand_recon:
                    kept_frames[t_idx] = 1
                    frame_data = hand_recon[frame_id]
                    
                    # Beta shape parameters (use first valid frame)
                    if not beta_set and 'beta' in frame_data:
                        beta = np.array(frame_data['beta']).astype(np.float64)
                        beta_set = True
                    
                    # Global orientation (wrist rotation)
                    # Use world-space if available (from _transform_to_world_space)
                    if 'global_orient_worldspace' in frame_data:
                        go_world = np.array(frame_data['global_orient_worldspace'])
                        if go_world.shape == (3, 3):
                            global_orient_worldspace[t_idx] = go_world.astype(np.float64)
                    
                    if 'global_orient_camspace' in frame_data:
                        go_cam = np.array(frame_data['global_orient_camspace'])
                        if go_cam.shape == (3, 3):
                            global_orient_camspace[t_idx] = go_cam.astype(np.float64)
                    elif 'global_orient' in frame_data:
                        # Fallback: use original global_orient as cam-space
                        go = np.array(frame_data['global_orient'])
                        if go.shape == (3, 3):
                            global_orient_camspace[t_idx] = go.astype(np.float64)
                            # If no world-space, use cam-space (static camera)
                            if 'global_orient_worldspace' not in frame_data:
                                global_orient_worldspace[t_idx] = go.astype(np.float64)
                    
                    # Hand pose (15 joint rotations) - same in cam and world space
                    if 'hand_pose' in frame_data:
                        hp = np.array(frame_data['hand_pose'])
                        # Expected shape: (15, 3, 3)
                        if hp.shape == (15, 3, 3):
                            hand_pose[t_idx] = hp.astype(np.float64)
                    
                    # Translation - use world-space if available
                    if 'transl_worldspace' in frame_data:
                        transl_world = np.array(frame_data['transl_worldspace']).flatten()[:3]
                        transl_worldspace[t_idx] = transl_world.astype(np.float64)
                        wrist[t_idx, 0] = transl_world.astype(np.float32)
                    
                    if 'transl_camspace' in frame_data:
                        transl_cam = np.array(frame_data['transl_camspace']).flatten()[:3]
                        transl_camspace[t_idx] = transl_cam.astype(np.float64)
                    elif 'transl' in frame_data:
                        # Fallback: use original transl as cam-space
                        transl = np.array(frame_data['transl']).flatten()[:3]
                        transl_camspace[t_idx] = transl.astype(np.float64)
                        # If no world-space, use cam-space (static camera)
                        if 'transl_worldspace' not in frame_data:
                            transl_worldspace[t_idx] = transl.astype(np.float64)
                            wrist[t_idx, 0] = transl.astype(np.float32)
            
            # Compute hand joints using MANO forward kinematics (simplified placeholder)
            # In a full implementation, this would use the MANO model
            for t_idx in range(T):
                if kept_frames[t_idx]:
                    # Use wrist as joint 0, others are relative offsets (placeholder)
                    joints_camspace[t_idx, 0] = wrist[t_idx, 0]
                    joints_worldspace[t_idx, 0] = wrist[t_idx, 0].astype(np.float64)
            
            # Compute movement statistics
            valid_indices = np.where(kept_frames == 1)[0]
            
            # Max translation movement
            max_translation_movement = None
            if len(valid_indices) > 1:
                translations = transl_worldspace[valid_indices]
                diffs = np.diff(translations, axis=0)
                movement_per_frame = np.linalg.norm(diffs, axis=1)
                max_translation_movement = float(np.max(movement_per_frame)) if len(movement_per_frame) > 0 else 0.0
            
            # Max wrist rotation movement
            max_wrist_rotation_movement = None
            if len(valid_indices) > 1:
                rotations = global_orient_worldspace[valid_indices]
                angle_diffs = []
                for i in range(len(rotations) - 1):
                    R_diff = rotations[i].T @ rotations[i+1]
                    # Compute rotation angle from rotation matrix
                    trace = np.trace(R_diff)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    angle_diffs.append(angle)
                max_wrist_rotation_movement = float(np.max(angle_diffs)) if angle_diffs else 0.0
            
            # Max finger joint angle movement
            max_finger_joint_angle_movement = None
            if len(valid_indices) > 1:
                poses = hand_pose[valid_indices]
                max_angle = 0.0
                for i in range(len(poses) - 1):
                    for j in range(15):
                        R_diff = poses[i, j].T @ poses[i+1, j]
                        trace = np.trace(R_diff)
                        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                        max_angle = max(max_angle, angle)
                max_finger_joint_angle_movement = float(max_angle)
            
            return {
                'beta': beta,
                'global_orient_camspace': global_orient_camspace,
                'global_orient_worldspace': global_orient_worldspace,
                'hand_pose': hand_pose,
                'transl_camspace': transl_camspace,
                'transl_worldspace': transl_worldspace,
                'kept_frames': kept_frames,
                'joints_camspace': joints_camspace,
                'joints_worldspace': joints_worldspace,
                'wrist': wrist,
                'max_translation_movement': max_translation_movement,
                'max_wrist_rotation_movement': max_wrist_rotation_movement,
                'max_finger_joint_angle_movement': max_finger_joint_angle_movement,
            }
        
        # Extract data for both hands
        left_data = extract_hand_data('left')
        right_data = extract_hand_data('right')
        
        # Compute global statistics
        anno_type = ep_info.get("anno_type", "right")
        primary_hand_data = left_data if anno_type == "left" else right_data
        
        # Avg speed (average wrist movement per frame)
        valid_indices = np.where(primary_hand_data['kept_frames'] == 1)[0]
        avg_speed = 0.0
        if len(valid_indices) > 1:
            translations = primary_hand_data['transl_worldspace'][valid_indices]
            diffs = np.diff(translations, axis=0)
            total_dist = np.sum(np.linalg.norm(diffs, axis=1))
            avg_speed = total_dist / (len(valid_indices) - 1)
        
        # Total rotation (camera rotation over episode) - placeholder using hand rotation
        total_rotvec_degree = 0.0
        if len(valid_indices) > 1:
            rotations = primary_hand_data['global_orient_worldspace'][valid_indices]
            total_angle = 0.0
            for i in range(len(rotations) - 1):
                R_diff = rotations[i].T @ rotations[i+1]
                trace = np.trace(R_diff)
                angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                total_angle += angle
            total_rotvec_degree = np.degrees(total_angle)
        
        # Total translation distance
        total_transl_dist = 0.0
        if len(valid_indices) > 1:
            translations = primary_hand_data['transl_worldspace'][valid_indices]
            diffs = np.diff(translations, axis=0)
            total_transl_dist = np.sum(np.linalg.norm(diffs, axis=1))
        
        # Build the final dictionary
        episode_dict = {
            # Clip segment info (deprecated but required for compatibility)
            'video_clip_id_segment': np.zeros(T, dtype=np.int64),
            
            # Camera parameters
            'extrinsics': extrinsics,
            'intrinsics': intrinsics_arr,
            
            # Frame info
            'video_decode_frame': np.array(frame_indices, dtype=np.int64),
            'video_name': ep_info.get("video_name", "unknown"),
            
            # Episode statistics
            'avg_speed': np.float64(avg_speed),
            'total_rotvec_degree': np.float64(total_rotvec_degree),
            'total_transl_dist': np.float64(total_transl_dist),
            
            # Annotation info
            'anno_type': anno_type,
            'text': ep_info.get("text", {"left": [], "right": []}),
            'text_rephrase': ep_info.get("text_rephrase", {"left": [], "right": []}),
            
            # Hand data
            'left': left_data,
            'right': right_data,
        }
        
        return episode_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VITRA Dataset Pipeline")
    parser.add_argument("--config", type=str, help="Path to config yaml")
    parser.add_argument("--input", type=str, required=True, help="Input video directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(args.config)
    try:
        builder.process_directory(args.input, args.output)
    finally:
        builder.pipleline_cleanup()
