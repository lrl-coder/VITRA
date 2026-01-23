"""
Atomic Action Segmentation Stage.
Segments continuous video into atomic action episodes based on motion heuristics.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass

from .base import TimedStage
from ..config import SegmentationConfig


@dataclass
class EpisodeSegment:
    """Represents a segmented atomic action."""
    start_frame: int
    end_frame: int
    hand_type: str  # 'left', 'right', or 'bimanual'
    score: float



class ActionSegmentationStage(TimedStage):
    """
    Stage 2: Atomic Action Segmentation.
    
    Implements the "Speed Minima" algorithm described in the VITRA paper.
    
    Algorithm:
    1. Input: Dense 3D wrist trajectories in world space.
    2. Missing Data Handling: Interpolate missing frames in trajectories.
    3. Smoothing: Apply Gaussian smoothing to the 3D positions to reduce jitter.
    4. Velocity Calculation: Compute the magnitude of the 3D velocity vector (speed).
    5. Minima Detection: Find local minima in the speed profile within a fixed time window (e.g., 0.5s).
    6. Segmentation: Use these minima as split points (start/end) for atomic actions.
    7. Independent Processing: Left and right hands are segmented independently.
    """
    
    def __init__(self, config: SegmentationConfig, logger=None):
        super().__init__(config, logger)

    def _do_initialize(self) -> None:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import argrelextrema
        self.gaussian_filter1d = gaussian_filter1d
        self.argrelextrema = argrelextrema

    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Segment the reconstructed data into episodes.
        """
        recon_data = input_data.get("reconstruction")
        if not recon_data:
            raise ValueError("No reconstruction data found for segmentation")
        
        # Get FPS from input data or config default
        fps = input_data.get("fps", self.config.fps)
        total_frames = input_data.get("total_frames", 0)
        
        # 1. Extract trajectories (filling missing data)
        # We need continuous trajectories for smoothing and derivative calculation
        left_traj = self._extract_trajectory(recon_data, "left", total_frames)
        right_traj = self._extract_trajectory(recon_data, "right", total_frames)
        
        # 2. Segment each hand independently
        left_segments = self._segment_hand_trajectory(left_traj, fps)
        right_segments = self._segment_hand_trajectory(right_traj, fps)
        
        episodes = []
        
        # Process Left Hand Segments
        for start, end in left_segments:
            episodes.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "anno_type": "left",
                "score": 1.0 # Placeholder
            })
            
        # Process Right Hand Segments
        for start, end in right_segments:
            episodes.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "anno_type": "right",
                "score": 1.0 # Placeholder
            })
        
        # Sort by start frame
        episodes.sort(key=lambda x: x["start_frame"])
        
        self.logger.info(f"Segmented {len(episodes)} atomic actions (L:{len(left_segments)}, R:{len(right_segments)}).")
        
        return {
            **input_data,
            "episodes": episodes
        }

    def _extract_trajectory(self, recon: Dict, hand: str, length: int) -> np.ndarray:
        """
        Extract translation trajectory as a (T, 3) array.
        Handles missing frames by linear interpolation.
        """
        traj = np.full((length, 3), np.nan)
        
        hand_data = recon.get(hand, {})
        
        # Fill known values
        # recon keys are frame indices
        for frame_idx, data in hand_data.items():
            if isinstance(frame_idx, int) and 0 <= frame_idx < length:
                t = data.get('transl') # 3D translation in world space
                if t is not None:
                    # Ensure it's numpy array
                    if isinstance(t, list): t = np.array(t)
                    if isinstance(t, torch.Tensor): t = t.cpu().numpy()
                    traj[frame_idx] = t
        
        # Interpolate missing values
        # Check each dimension independently
        x = np.arange(length)
        for i in range(3):
            y = traj[:, i]
            mask = ~np.isnan(y)
            if np.any(mask):
                # Interpolate using known values
                y_interp = np.interp(x, x[mask], y[mask])
                traj[:, i] = y_interp
            else:
                # If absolutely no data for this dimension, fill with 0
                traj[:, i] = 0.0
                
        return traj

    def _segment_hand_trajectory(self, trajectory: np.ndarray, fps: float) -> List[Tuple[int, int]]:
        """
        Implements the speed minima segmentation algorithm.
        """
        T = len(trajectory)
        if T < 2:
            return []
            
        # 1. Smoothing
        # Apply Gaussian smoothing to positions to reduce jitter
        # Sigma controls simple smoothing strength
        smoothed_traj = self.gaussian_filter1d(trajectory, sigma=self.config.smoothing_sigma, axis=0)
        
        # 2. Velocity Calculation (Speed)
        # Compute gradient (velocity vector)
        vel_vector = np.gradient(smoothed_traj, axis=0) # (T, 3)
        # Compute magnitude (speed scalar)
        speed = np.linalg.norm(vel_vector, axis=1) # (T,)
        
        # Further smooth the speed profile itself if needed, but position smoothing is usually enough.
        # Let's align with the paper: "Speed minima of the 3D hand wrists"
        
        # 3. Minima Detection
        # We want local minima within a window.
        # Window size in frames
        window_frames = int(self.config.minima_window_size * fps)
        
        # Use argrelextrema to find local minima
        # order is the number of points on each side to compare
        # order = window_frames // 2 seems reasonable to represent the window
        order = max(1, window_frames // 2)
        
        minima_indices = self.argrelextrema(speed, np.less, order=order)[0]
        
        # Add start and end frames as implicit boundaries if they are not included
        # Though the paper says "use these points as split points".
        # Depending on trajectory, start/end might be idle (zero speed), so they are naturally minima.
        # Let's enforce 0 and T-1 as boundaries to capture the first and last actions.
        boundaries = [0] + list(minima_indices) + [T-1]
        boundaries = sorted(list(set(boundaries)))
        
        # 4. Generate Segments
        segments = []
        min_len = int(self.config.min_segment_duration * fps)
        max_len = int(self.config.max_segment_duration * fps)
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            duration = end - start
            
            # Simple filtering logic
            if duration < min_len:
                continue
            
            # If segment is too long, it might be holding/idle or a long motion without clear minima.
            # For "Atomic" actions, extremely long segments usually imply we missed a split
            # or it is a long idle period. 
            # The paper implies simple splitting. We will keep it but maybe warn or flag.
            if duration > max_len:
                 # Optional: could split equally or discard. For now, keep it.
                 pass
                 
            segments.append((start, end))
            
        return segments
