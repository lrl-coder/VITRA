"""
Video processing utilities for the dataset construction pipeline.
Handles video loading, frame extraction, and preprocessing.
"""

import os
from typing import List, Tuple, Optional, Generator, Dict, Any
from pathlib import Path
import logging

import numpy as np
import cv2
from tqdm import tqdm

from .base import TimedStage, StageResult
from ..config import VideoConfig


class VideoProcessor(TimedStage):
    """
    Processes videos for the dataset construction pipeline.
    
    Handles:
    - Video loading and frame extraction
    - Resolution normalization
    - Fisheye undistortion (optional)
    - Frame batching for efficient processing
    """
    
    def __init__(self, config: VideoConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.intrinsics_cache: Dict[str, np.ndarray] = {}
    
    def _do_initialize(self) -> None:
        """Load intrinsics if undistortion is enabled."""
        if self.config.apply_undistortion and self.config.intrinsics_path:
            self._load_intrinsics()
    
    def _load_intrinsics(self) -> None:
        """Load camera intrinsics from file."""
        intrinsics_path = Path(self.config.intrinsics_path)
        if intrinsics_path.is_dir():
            for f in intrinsics_path.glob("*.npy"):
                video_name = f.stem
                self.intrinsics_cache[video_name] = np.load(f, allow_pickle=True)
            self.logger.info(f"Loaded intrinsics for {len(self.intrinsics_cache)} videos")
    
    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            input_data: Dictionary containing:
                - video_path: Path to the video file
                - output_dir: Optional output directory for processed frames
                
        Returns:
            Dictionary containing:
                - frames: List of processed frames (numpy arrays)
                - frame_indices: Original frame indices
                - fps: Video FPS
                - total_frames: Total number of frames
                - intrinsics: Camera intrinsics (if available)
        """
        video_path = Path(input_data["video_path"])
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Extract frames
        frames, metadata = self.extract_frames(video_path)
        
        # Apply preprocessing
        if self.config.resize_short_side:
            frames = [self._resize_frame(f) for f in frames]
        
        if self.config.apply_undistortion:
            video_name = video_path.stem
            if video_name in self.intrinsics_cache:
                frames = self._undistort_frames(frames, self.intrinsics_cache[video_name])
        
        return {
            "frames": frames,
            "frame_indices": list(range(len(frames))),
            "fps": metadata["fps"],
            "total_frames": metadata["total_frames"],
            "width": metadata["width"],
            "height": metadata["height"],
            "intrinsics": self.intrinsics_cache.get(video_path.stem),
        }
    
    def extract_frames(
        self,
        video_path: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            step: Frame step (1 = every frame, 2 = every other frame, etc.)
            
        Returns:
            Tuple of (frames list, metadata dict)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if end_frame is None:
            end_frame = total_frames
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, min(end_frame, total_frames), step):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
            # Skip frames if step > 1
            if step > 1:
                for _ in range(step - 1):
                    cap.read()
        
        cap.release()
        
        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
        }
        
        self.logger.debug(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames, metadata
    
    def extract_frames_generator(
        self,
        video_path: Path,
        batch_size: int = 100
    ) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
        """
        Generator that yields batches of frames for memory-efficient processing.
        
        Args:
            video_path: Path to the video file
            batch_size: Number of frames per batch
            
        Yields:
            Tuple of (frames batch, frame indices)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        batch = []
        indices = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            batch.append(frame)
            indices.append(frame_idx)
            frame_idx += 1
            
            if len(batch) >= batch_size:
                yield batch, indices
                batch = []
                indices = []
        
        # Yield remaining frames
        if batch:
            yield batch, indices
        
        cap.release()
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame keeping aspect ratio."""
        h, w = frame.shape[:2]
        target = self.config.resize_short_side
        
        if h < w:
            new_h = target
            new_w = int(w * target / h)
        else:
            new_w = target
            new_h = int(h * target / w)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def _undistort_frames(
        self,
        frames: List[np.ndarray],
        intrinsics: np.ndarray
    ) -> List[np.ndarray]:
        """
        Apply fisheye undistortion to frames.
        
        Args:
            frames: List of frames to undistort
            intrinsics: Camera intrinsics array containing distortion parameters
            
        Returns:
            List of undistorted frames
        """
        # Parse intrinsics (format depends on source)
        if isinstance(intrinsics, dict):
            K = intrinsics.get("camera_matrix", intrinsics.get("K"))
            dist_coeffs = intrinsics.get("dist_coeffs", intrinsics.get("D"))
        else:
            # Assume intrinsics is a numpy array with specific format
            K = intrinsics[:3, :3] if intrinsics.ndim == 2 else intrinsics
            dist_coeffs = None
        
        if dist_coeffs is None:
            self.logger.warning("No distortion coefficients found, skipping undistortion")
            return frames
        
        h, w = frames[0].shape[:2]
        
        # Compute undistortion maps once
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
        
        undistorted = []
        for frame in frames:
            undist_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            undistorted.append(undist_frame)
        
        return undistorted
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get video metadata without loading frames."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        return info
    
    @staticmethod
    def find_videos(
        directory: Path,
        formats: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Find all video files in a directory.
        
        Args:
            directory: Directory to search
            formats: List of video formats to include (e.g., [".mp4", ".avi"])
            
        Returns:
            List of video file paths
        """
        if formats is None:
            formats = [".mp4", ".MP4", ".avi", ".webm", ".mov"]
        
        videos = []
        for fmt in formats:
            videos.extend(directory.glob(f"**/*{fmt}"))
        
        return sorted(videos)
