"""
Visualization Stage.
Renders the processing results (hand pose, segmentation, text) onto the video.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
# Fix potential cv2/gl issue
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base import TimedStage
from ..config import OutputConfig

try:
    from visualization.visualize_core import HandConfig, HandVisualizer, Renderer, process_single_hand_labels
    import torch
    HAS_3D_VIS = True
except ImportError:
    HAS_3D_VIS = False
    print("Warning: Visualization core or pytorch3d not found. 3D visualization will be disabled.")


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
        self.enable_3d = False
        if HAS_3D_VIS:
            try:
                # Setup configuration for HandVisualizer
                class Args: pass
                args = Args()
                args.mano_model_path = "weights/mano" # Ensure this path is correct or configurable
                
                self.hand_config = HandConfig(args)
                self.hand_config.FPS = getattr(self.config, 'fps', 30)
                
                # Check for MANO weights
                if not Path(args.mano_model_path).exists():
                    self.logger.warning(f"MANO weights not found at {args.mano_model_path}. 3D visualization disabled.")
                else:
                    self.visualizer = HandVisualizer(self.hand_config, render_gradual_traj=False)
                    self.enable_3d = True
                    self.logger.info("3D Hand Visualization initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize 3D visualizer: {e}")
                import traceback
                self.logger.error(traceback.format_exc())


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
            
            # Append if not 3D (logic handled later)
            # out.write(vis_frame) # Moved out

        # Perform 3D Rendering Batch if enabled
        final_frames = frames 
        if self.enable_3d and recon:
            self.logger.info("Rendering 3D hand meshes...")
            try:
                final_frames = self._render_3d_batch(frames, recon, episodes)
            except Exception as e:
                self.logger.error(f"3D Rendering failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                final_frames = frames

        # Write frames to video
        for i, frame in enumerate(final_frames):
            # If 3D rendering was done, frame is already processed. 
            # But we still need to add HUD if it wasn't added during 3D pass.
            # Actually, let's keep HUD drawing on the 'vis_frame' before 3D if possible, 
            # or draw HUD on top of 3D. 
            # _render_3d_batch returns new frames (RGB). 
            # Let's assume we want HUD on top.
            
            vis_frame = frame.copy()
            if self.enable_3d:
                # If 3D enabled, frame is RGB from renderer, convert to BGR for OpenCV
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            
            # HUD
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

    def _render_3d_batch(self, frames: List[np.ndarray], recon: Dict[str, Any], episodes: List[Dict]) -> List[np.ndarray]:
        """
        Render 3D hand meshes for the entire video sequence using HandVisualizer.
        """
        T = len(frames)
        h, w = frames[0].shape[:2]
        
        # 1. Prepare Trajectory Data
        # We need to construct continuous arrays for process_single_hand_labels
        
        # helper to extract data
        def extract_hand_data(hand_side):
            # Initialize arrays
            transl = np.zeros((T, 3), dtype=np.float32)
            global_orient = np.eye(3).reshape(1, 3, 3).repeat(T, axis=0) # (T, 3, 3)
            hand_pose = np.eye(3).reshape(1, 15, 3, 3).repeat(T, axis=0) # (T, 15, 3, 3)
            beta = np.zeros(10, dtype=np.float32)
            mask = np.zeros(T, dtype=bool)
            
            hand_recon = recon.get(hand_side, {})
            # hand_recon is frame_idx -> dict
            
            first_beta_found = False
            
            for idx in range(T):
                if idx in hand_recon:
                    data = hand_recon[idx]
                    
                    if 'transl' in data:
                        transl[idx] = data['transl']
                    
                    if 'global_orient' in data:
                        go = data['global_orient']
                        # format could be matrix (3,3)
                        if go.shape == (3, 3):
                            global_orient[idx] = go
                        else:
                            # assume euler?
                            pass
                            
                    if 'hand_pose' in data:
                        hp = data['hand_pose']
                        # format could be (15, 3, 3) or (45,) or (15, 3)
                        if hp.ndim == 3 and hp.shape == (15, 3, 3):
                            hand_pose[idx] = hp
                        # If needed add other format handling
                        
                    if 'beta' in data and not first_beta_found:
                        beta = data['beta']
                        first_beta_found = True
                        
                    mask[idx] = True
            
            return {
                'transl_worldspace': transl,
                'global_orient_worldspace': global_orient,
                'hand_pose': hand_pose,
                'beta': beta
            }, mask

        left_labels, left_mask = extract_hand_data('left')
        right_labels, right_mask = extract_hand_data('right')

        # 2. Process Hand Labels to Vertices
        # We assume recon 'transl' is in CAMERA space (which we treat as world for visualization with Identity extrinsics)
        # But wait, process_single_hand_labels does: (verts - J0) + transl
        # So if transl is camera space coordinates, we are good.
        
        verts_left, _ = process_single_hand_labels(left_labels, left_mask, self.visualizer.mano, is_left=True)
        verts_right, _ = process_single_hand_labels(right_labels, right_mask, self.visualizer.mano, is_left=False)
        
        hand_traj_worldspace = (verts_left, verts_right)
        hand_mask = (left_mask, right_mask)
        
        # 3. Setup Renderer
        # We need intrinsics. recon['intrinsics'] should be there?
        # If not, guess from FOV or provide default
        fx_exo, fy_exo = 1000.0, 1000.0 # fallback
        if 'intrinsics' in recon:
             K = recon['intrinsics']
             fx_exo = K[0, 0]
             fy_exo = K[1, 1]
        elif 'fov_x' in recon:
             fov_x = recon['fov_x']
             # fov_x is likely degrees?
             # f = w / (2 * tan(fov/2))
             f = w / (2 * np.tan(np.deg2rad(fov_x) / 2))
             fx_exo = f
             fy_exo = f
        
        renderer = Renderer(w, h, (fx_exo, fy_exo), 'cuda')
        
        # 4. Extrinsics (World to Camera)
        # Since our "world" is actually camera space, extrinsics are Identity
        R_w2c = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        t_w2c = np.zeros((T, 3, 1), dtype=np.float32)
        extrinsics = (R_w2c, t_w2c)
        
        # 5. Render
        # frames must be BGR (cv2), convert to RGB for renderer?
        # Renderer checks usage in visualize_core.py: 
        # "curr_img_overlay = video_frames[current_frame_idx].copy().astype(np.float32) / 255.0"
        # Since we loaded with cv2 (BGR) in video_processor, but VisualizationStage gets them as BGR.
        # However, visualize_core.py imports cv2 and likely expects BGR or RGB?
        # In visualize_core: "final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)" at the end.
        # This implies it produces RGB.
        # And it calls "rend = rend[..., ::-1] # RGB to BGR" inside loop.
        # So input should be BGR if we want colors correct?
        # Actually inference_human_prediction.py: "image_bgr = image_resized_np[..., ::-1]" -> creates BGR.
        
        vis_frames = [f.copy() for f in frames] # Make copies
        
        rendered_frames = self.visualizer._render_hand_trajectory(
            vis_frames,
            hand_traj_wordspace,
            hand_mask,
            extrinsics,
            renderer,
            mode='cam' # Render current frame mesh
        )
        
        return rendered_frames
