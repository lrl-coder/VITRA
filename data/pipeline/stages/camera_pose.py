"""
Camera Pose Estimation Stage.
Estimates camera poses using DROID-SLAM for ego-centric videos.
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np
import torch
from pathlib import Path

from .base import TimedStage
from ..config import PipelineConfig

# Add HaWoR path for SLAM utilities
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
hawor_path = os.path.join(project_root, "thirdparty", "HaWoR")

class CameraPoseEstimationStage(TimedStage):
    """
    Stage for estimating camera poses from video frames.
    
    This stage uses DROID-SLAM (masked version) to track camera motion
    and outputs per-frame camera extrinsics (world-to-camera transforms).
    
    The camera poses are essential for:
    1. Converting hand poses from camera-space to world-space
    2. Generating stable, world-fixed action sequences
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.slam_available = False
        self.device = None
        
    def _do_initialize(self) -> None:
        """Initialize SLAM components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to import DROID-SLAM
        try:
            droid_slam_path = os.path.join(hawor_path, "thirdparty", "DROID-SLAM", "droid_slam")
            if droid_slam_path not in sys.path:
                sys.path.insert(0, droid_slam_path)
                sys.path.insert(0, os.path.join(hawor_path, "thirdparty", "DROID-SLAM"))
            
            from droid import Droid
            self.slam_available = True
            self.logger.info("DROID-SLAM initialized successfully")
        except ImportError as e:
            self.logger.warning(f"DROID-SLAM not available: {e}")
            self.logger.warning("Camera poses will use identity (static camera assumption)")
            self.slam_available = False
    
    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate camera poses for all frames.
        
        Args:
            input_data: Dictionary containing:
                - frames: List of video frames (numpy arrays, BGR)
                - intrinsics: Camera intrinsics (if available)
                - reconstruction: Hand reconstruction results (for masking)
                
        Returns:
            input_data with added:
                - camera_poses: Dict with R_c2w, t_c2w, R_w2c, t_w2c arrays
        """
        frames = input_data.get("frames", [])
        T = len(frames)
        
        if T == 0:
            return input_data
        
        h, w = frames[0].shape[:2]
        
        # Get or compute focal length
        intrinsics = input_data.get("intrinsics")
        recon = input_data.get("reconstruction", {})
        
        if intrinsics is not None:
            focal = intrinsics[0, 0]  # fx from intrinsics matrix
        elif "fov_x" in recon:
            fov_x = recon["fov_x"]
            focal = w / (2 * np.tan(np.deg2rad(fov_x) / 2))
        else:
            # Default assumption
            focal = max(h, w)
        
        if self.slam_available:
            try:
                camera_poses = self._run_slam(frames, focal, recon)
                self.logger.info(f"SLAM completed: {T} camera poses estimated")
            except Exception as e:
                self.logger.error(f"SLAM failed: {e}, using identity poses")
                camera_poses = self._get_identity_poses(T)
        else:
            self.logger.info("Using identity camera poses (static camera assumption)")
            camera_poses = self._get_identity_poses(T)
        
        input_data["camera_poses"] = camera_poses
        return input_data
    
    def _run_slam(self, frames: list, focal: float, recon: Dict) -> Dict[str, np.ndarray]:
        """
        Run DROID-SLAM on the video frames.
        
        Args:
            frames: List of video frames (BGR numpy arrays)
            focal: Estimated focal length
            recon: Hand reconstruction results (for generating hand masks)
            
        Returns:
            Dictionary with camera pose arrays
        """
        import tempfile
        import cv2
        
        T = len(frames)
        h, w = frames[0].shape[:2]
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save frames as images
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), frame)
            
            # Prepare calibration
            cx, cy = w / 2, h / 2
            calib = [focal, focal, cx, cy]
            
            # Generate hand masks for masked SLAM (improves accuracy)
            masks = self._generate_hand_masks(frames, recon, h, w)
            
            # Run SLAM
            try:
                from lib.pipeline.masked_droid_slam import run_slam
                droid, traj = run_slam(
                    tmpdir, 
                    masks, 
                    calib=calib,
                    stride=1,
                    filter_thresh=2.4,
                    disable_vis=True
                )
                
                # Extract camera poses from trajectory
                # traj format: [N, 7] where each row is [tx, ty, tz, qw, qx, qy, qz]
                if traj is not None and len(traj) > 0:
                    return self._trajectory_to_poses(traj, T)
                    
            except Exception as e:
                self.logger.warning(f"Masked SLAM failed: {e}, trying simple SLAM")
                
                try:
                    from lib.pipeline.masked_droid_slam import run_droid_slam
                    droid, traj = run_droid_slam(
                        tmpdir,
                        calib=calib,
                        stride=1,
                        disable_vis=True
                    )
                    
                    if traj is not None and len(traj) > 0:
                        return self._trajectory_to_poses(traj, T)
                        
                except Exception as e2:
                    self.logger.error(f"Simple SLAM also failed: {e2}")
        
        # Fallback to identity
        return self._get_identity_poses(T)
    
    def _generate_hand_masks(self, frames: list, recon: Dict, h: int, w: int) -> torch.Tensor:
        """
        Generate hand masks for masked SLAM.
        
        Masked SLAM ignores regions with moving hands/objects to improve
        camera tracking accuracy.
        """
        T = len(frames)
        masks = torch.zeros((T, 1, h, w), dtype=torch.float32)
        
        # If no reconstruction data, return empty masks
        if not recon:
            return masks
        
        # TODO: Generate actual hand masks from reconstruction
        # For now, return empty masks (no masking)
        # In full implementation, would render hand meshes and use as masks
        
        return masks
    
    def _trajectory_to_poses(self, traj: np.ndarray, T: int) -> Dict[str, np.ndarray]:
        """
        Convert SLAM trajectory to camera pose matrices.
        
        Args:
            traj: SLAM output [N, 7] with [tx, ty, tz, qw, qx, qy, qz]
            T: Number of frames
            
        Returns:
            Dictionary with R_c2w, t_c2w, R_w2c, t_w2c arrays
        """
        N = len(traj)
        
        # Handle different trajectory lengths
        if N < T:
            # Interpolate or pad
            self.logger.warning(f"SLAM trajectory ({N}) shorter than frames ({T}), padding")
            pad = np.tile(traj[-1:], (T - N, 1))
            traj = np.vstack([traj, pad])
        elif N > T:
            traj = traj[:T]
        
        # Extract translation and quaternion
        t_c2w = traj[:, :3]  # (T, 3)
        q_c2w = traj[:, 3:]  # (T, 4) - [qw, qx, qy, qz] or [qx, qy, qz, qw]
        
        # Convert quaternion to rotation matrix
        R_c2w = self._quaternion_to_matrix(q_c2w)  # (T, 3, 3)
        
        # Compute inverse (world to camera)
        R_w2c = np.transpose(R_c2w, (0, 2, 1))  # (T, 3, 3)
        t_w2c = -np.einsum("tij,tj->ti", R_w2c, t_c2w)  # (T, 3)
        
        # Build 4x4 extrinsics matrix (world to camera)
        extrinsics = np.zeros((T, 4, 4), dtype=np.float64)
        extrinsics[:, :3, :3] = R_w2c
        extrinsics[:, :3, 3] = t_w2c
        extrinsics[:, 3, 3] = 1.0
        
        return {
            "R_c2w": R_c2w.astype(np.float64),
            "t_c2w": t_c2w.astype(np.float64),
            "R_w2c": R_w2c.astype(np.float64),
            "t_w2c": t_w2c.astype(np.float64),
            "extrinsics": extrinsics,
        }
    
    def _quaternion_to_matrix(self, quaternions: np.ndarray) -> np.ndarray:
        """
        Convert quaternions to rotation matrices.
        
        Args:
            quaternions: Array of shape (N, 4) with [qw, qx, qy, qz] or [qx, qy, qz, qw]
            
        Returns:
            Rotation matrices of shape (N, 3, 3)
        """
        # Assume input is [qx, qy, qz, qw] (DROID-SLAM format)
        # Convert to [qw, qx, qy, qz]
        q = quaternions.copy()
        if q.shape[1] == 4:
            # Check if it looks like xyzw format (common if w is at end)
            # For normalized quaternions, w should typically be the largest
            # This is a heuristic - in practice, need to know the format
            q = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Normalize
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        N = len(q)
        R = np.zeros((N, 3, 3))
        
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return R
    
    def _get_identity_poses(self, T: int) -> Dict[str, np.ndarray]:
        """
        Return identity camera poses (static camera assumption).
        
        When SLAM is not available or fails, we assume the camera is static.
        This means camera-space == world-space.
        """
        R_identity = np.eye(3, dtype=np.float64)
        t_zero = np.zeros(3, dtype=np.float64)
        
        extrinsics = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
        
        return {
            "R_c2w": np.tile(R_identity, (T, 1, 1)),
            "t_c2w": np.tile(t_zero, (T, 1)),
            "R_w2c": np.tile(R_identity, (T, 1, 1)),
            "t_w2c": np.tile(t_zero, (T, 1)),
            "extrinsics": extrinsics,
        }
