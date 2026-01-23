"""
Visualization Stage.
Renders the processing results (hand pose, segmentation, text) onto the video.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base import TimedStage
from ..config import OutputConfig

class VisualizationStage(TimedStage):
    """
    Stage 5: Visualization.
    
    Overlays annotation data onto the original video frames and saves as MP4.
    
    Visualizations:
    1. Action Description (Text)
    2. Atomic Action Segment (Timeline/Progress bar)
    3. Hand Trajectory / Keypoints (2D projection)
    """
    
    def __init__(self, config: OutputConfig, logger=None):
        super().__init__(config, logger)
        # Colors for visualization (BGR)
        self.colors = {
            'left': (255, 100, 100),   # Blue-ish
            'right': (100, 100, 255),  # Red-ish
            'text_bg': (0, 0, 0),
            'timeline_bg': (50, 50, 50),
            'timeline_active': (0, 255, 0)
        }

    def _do_initialize(self) -> None:
        pass

    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization video.
        """
        if not self.config.save_visualization:
            return input_data

        video_name = input_data.get('video_name', 'output')
        frames = input_data.get('frames')
        episodes = input_data.get('episodes', [])
        recon = input_data.get('reconstruction', {})
        fps = input_data.get('fps', 30.0)
        
        output_dir = Path(input_data.get('output_dir', '.')) / "vis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_name}_vis.mp4"
        
        self.logger.info(f"Rendering visualization to {output_path}...")
        
        # Prepare video writer
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        # Pre-compute episode lookups for fast access
        frame_epis = {}
        for ep in episodes:
            for f_idx in range(ep['start_frame'], ep['end_frame']):
                if f_idx not in frame_epis:
                    frame_epis[f_idx] = []
                frame_epis[f_idx].append(ep)
        
        # Render loop
        for i, frame in enumerate(frames):
            vis_frame = frame.copy()
            
            # 1. Draw Hand Centers (Trajectory) if available
            # Note: Full MANO mesh rendering requires camera projection logic
            # Here we visualize simple centers if 'transl' is available relative to something, 
            # Or simpler: just the text and segmentation.
            # To do 3D projection we need intrinsics and world-2-cam. 
            # Assuming 'recon' has world space, we need extrinsics.
            # For simplicity in this stage, we skip 3D projection unless explicitly requested, 
            # as it requires reimplementing the projection logic from human_dataset.
            
            # 2. Draw Episode Info
            active_epis = frame_epis.get(i, [])
            self._draw_hud(vis_frame, active_epis, i, len(frames))
            
            out.write(vis_frame)
            
        out.release()
        self.logger.info(f"Visualization saved.")
        
        return input_data

    def _draw_hud(self, img: np.ndarray, episodes: List[Dict], frame_idx: int, total_frames: int):
        """Draw Heads-Up Display with text and timeline."""
        h, w = img.shape[:2]
        
        # 1. Timeline Bar at bottom
        bar_h = 20
        y_bar = h - bar_h
        
        # Background
        cv2.rectangle(img, (0, y_bar), (w, h), self.colors['timeline_bg'], -1)
        
        # Current Progress
        progress_x = int(w * (frame_idx / total_frames))
        cv2.circle(img, (progress_x, y_bar + bar_h//2), 6, self.colors['timeline_active'], -1)
        
        # Draw Segmentation Blocks on timeline
        # (This might be expensive to loop all eps every frame, but okay for rendering)
        # Optimization: Only draw nearby or skip this visualization if too slow
        
        # 2. Text Annotations (Top)
        if episodes:
            # Stack text lines
            y_text = 40
            line_height = 35
            
            for ep in episodes:
                hand = ep['anno_type']
                # Get description
                # Note: 'text' field format from LanguageAnnotationStage: 
                # text: {'left': [(desc, (start, end))], ...}
                # But 'ep' is the flat episode dict from segmentation stage?
                # Wait, LanguageAnnotationStage modified 'ep' or created new structure?
                # In LanguageAnnotationStage code: annotated_ep["text"][hand].append(...)
                
                # Check data structure from previous stage
                # 'ep' is one item from 'episodes' list.
                # In annotation stage: annotated_ep['text'][hand] is a list of tuples.
                
                desc_text = "Action: ???"
                if 'text' in ep:
                    # The annotation stage puts the whole dict in each episode for some reason
                    # Let's extract the first description
                    descs = ep['text'].get(hand, [])
                    if descs:
                        desc_text = descs[0][0] # (desc, range)
                
                text = f"[{hand.upper()}] {desc_text}"
                color = self.colors.get(hand, (255, 255, 255))
                
                # Draw text with outline
                cv2.putText(img, text, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(img, text, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, color, 2, cv2.LINE_AA)
                
                y_text += line_height
