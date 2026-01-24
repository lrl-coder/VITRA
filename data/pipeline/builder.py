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
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.logger = logging.getLogger("DatasetBuilder")
        
        # Initialize Stages
        self.logger.info("Initializing pipeline stages...")
        self.video_processor = VideoProcessor(self.config.video, self.logger)
        self.hand_recon = HandReconstructionStage(self.config.hand_recon, self.logger)
        self.segmenter = ActionSegmentationStage(self.config.segmentation, self.logger)
        self.annotator = LanguageAnnotationStage(self.config.annotation, self.logger)
        self.visualizer = VisualizationStage(self.config.output, self.logger)
        
        # Initialize all stages (load models etc)
        self._init_stages()

    def _init_stages(self):
        self.video_processor.initialize()
        self.hand_recon.initialize()
        self.segmenter.initialize()
        self.annotator.initialize()
        self.visualizer.initialize()

    def pipleline_cleanup(self):
        self.video_processor.cleanup()
        self.hand_recon.cleanup()
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
        """
        self.logger.info(f"Starting processing for {video_path.name}")
        video_name = video_path.stem
        dataset_name = "custom" # Default dataset name
        
        # --- Stage 1: Video Processing ---
        video_data = self.video_processor.process({
            "video_path": video_path
        })
        
        # --- Stage 2: Hand Reconstruction ---
        recon_data = self.hand_recon.process(video_data)
        
        # --- Stage 3: Segmentation ---
        segmented_data = self.segmenter.process(recon_data)
        
        # --- Stage 4: Annotation ---
        final_data = self.annotator.process(segmented_data)
        
        # --- Stage 5: Visualization (Optional) ---
        # Pass output_root and video_name to visualization stage
        final_data['output_dir'] = output_root
        final_data['video_name'] = video_name
        self.visualizer.process(final_data)
        
        # --- Stage 6: Save Output ---
        self._save_results(final_data, output_root, dataset_name, video_name)
        
        self.logger.info(f"Completed {video_path.name}")

    def _save_results(self, data: Dict[str, Any], output_root: Path, dataset_name: str, video_name: str):
        """
        Save results in the format expected by VITRA human_dataset.py.
        
        Format:
        {dataset_name}_{video_name}_ep_{ep_idx}.npy
        """
        episodes = data.get("episodes", [])
        recon = data.get("reconstruction", {})
        intrinsics = data.get("intrinsics")
        
        # Ensure subdirectories
        save_dir = output_root / "episodic_annotations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert internal format to VITRA .npy format
        # See data/data.md section 4 for structure
        
        for i, ep in enumerate(episodes):
            ep_filename = f"{dataset_name}_{video_name}_ep_{i:06d}.npy"
            save_path = save_dir / ep_filename
            
            # Construct the complex dictionary structure VITRA expects
            # This is a reconstruction of the structure documented in data/data.md
            
            start = ep["start_frame"]
            end = ep["end_frame"]
            length = end - start
            
            # Slice reconstruction data for this episode
            # Note: The HandReconstructionStage output format needs to be mapped 
            # to the flat arrays expected by VITRA (.npy structure)
            
            # Placeholder for mapping logic:
            # VITRA expects: 
            # 'extrinsics': (T, 4, 4) -> Needs to be derived from camera pose
            # 'left': {'beta', 'global_orient_worldspace', 'hand_pose', ...}
            
            episode_dict = self._format_vitra_episode(ep, recon, intrinsics, start, end)
            
            np.save(save_path, episode_dict)
            
        # Update index file (optional, but needed for loading)
        # self._update_index_file(...)

    def _format_vitra_episode(self, ep_info, recon_full, intrinsics, start_idx, end_idx):
        """
        Helper to format a single episode slice into the VITRA dictionary format.
        
        The expected format is documented in data/data.md section 4.
        
        Args:
            ep_info: Episode information dict containing text, anno_type, etc.
            recon_full: Reconstruction results with 'left', 'right', 'fov_x' keys.
                        Each hand contains {frame_id: {beta, hand_pose, global_orient, transl}}
            intrinsics: Camera intrinsic matrix (3x3)
            start_idx: Start frame index
            end_idx: End frame index (exclusive)
            
        Returns:
            Dictionary in VITRA metadata format.
        """
        T = end_idx - start_idx
        frame_indices = list(range(start_idx, end_idx))
        
        # Build extrinsics array - using identity as placeholder
        # In reality, this should come from camera pose estimation
        extrinsics = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)
        
        # Build intrinsics - use provided or default
        if intrinsics is not None:
            intrinsics_arr = np.array(intrinsics).astype(np.float64)
        else:
            intrinsics_arr = np.eye(3).astype(np.float64)
        
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
                    if 'global_orient' in frame_data:
                        global_orient = np.array(frame_data['global_orient'])
                        # Ensure (3, 3) shape
                        if global_orient.shape == (3, 3):
                            global_orient_camspace[t_idx] = global_orient.astype(np.float64)
                            # World space = extrinsics inverse @ cam space
                            # For identity extrinsics, worldspace = camspace
                            global_orient_worldspace[t_idx] = global_orient.astype(np.float64)
                    
                    # Hand pose (15 joint rotations)
                    if 'hand_pose' in frame_data:
                        hp = np.array(frame_data['hand_pose'])
                        # Expected shape: (15, 3, 3)
                        if hp.shape == (15, 3, 3):
                            hand_pose[t_idx] = hp.astype(np.float64)
                    
                    # Translation
                    if 'transl' in frame_data:
                        transl = np.array(frame_data['transl']).flatten()[:3]
                        transl_camspace[t_idx] = transl.astype(np.float64)
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
