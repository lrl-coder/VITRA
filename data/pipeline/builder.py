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
        """
        # This function would need detailed mapping logic reading from 'recon_full'
        # which currently contains 'left' and 'right' dicts of frames.
        
        # Create Dummy/Placeholder Output that matches structure for now
        # to ensure Pipeline runs
        
        T = end_idx - start_idx
        
        def get_slice(hand_side):
            # Extract arrays from recon_full[hand_side] for range [start, end]
            # recon_full[hand_side] is assumed to be {frame_id: dict_of_params}
            # Need to stack them
            pass
        
        # Mock Extrinsics (Identity if not available)
        extrinsics = np.eye(4)[None].repeat(T, axis=0) 
        
        return {
            'video_name': ep_info.get("video_name", "unknown"),
            'anno_type': ep_info["anno_type"],
            'text': ep_info["text"],
            'text_rephrase': ep_info["text_rephrase"],
            'video_decode_frame': list(range(start_idx, end_idx)),
            'extrinsics': extrinsics,
            'intrinsics': intrinsics if intrinsics is not None else np.eye(3),
            # Add other keys as per data/data.md
            'left': {
                'beta': np.zeros(10), # Mock
                'kept_frames': np.ones(T, dtype=int), # Mock
                # ... other required fields ...
            },
            'right': {
                'beta': np.zeros(10),
                 'kept_frames': np.ones(T, dtype=int),
            }
        }

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
