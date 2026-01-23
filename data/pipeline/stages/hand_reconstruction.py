"""
Hand and Camera Pose Reconstruction Stage.
Wraps the existing hand_recon_core tools to perform 3D reconstruction.
"""

from typing import Dict, Any, List, Optional
import sys
import os
import torch
import numpy as np
from types import SimpleNamespace

# Add project root to path if needed for relative imports in legacy code
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from data.tools
try:
    from data.tools.hand_recon_core import HandReconstructor, Config as LegacyConfig
except ImportError:
    # Fallback to relative import if running as package
    from ...tools.hand_recon_core import HandReconstructor, Config as LegacyConfig

from .base import TimedStage
from ..config import HandReconConfig


class HandReconstructionStage(TimedStage):
    """
    Stage 1: Hand and Camera Pose Reconstruction.
    
    Uses MoGe for FOV estimation and HaWoR for hand tracking and pose estimation.
    Wraps the functionality provided in data/tools/hand_recon_core.py.
    """
    
    def __init__(self, config: HandReconConfig, logger=None):
        super().__init__(config, logger)
        self.reconstructor: Optional[HandReconstructor] = None
        self.device: Optional[torch.device] = None

    def _do_initialize(self) -> None:
        """Initialize the HandReconstructor with configuration."""
        self.logger.info("Initializing HandReconstructor...")
        
        # The legacy Config expects an args object with attributes
        args = SimpleNamespace(
            hawor_model_path=self.config.hawor_model_path,
            detector_path=self.config.detector_path,
            moge_model_path=self.config.moge_model_path,
            mano_path=self.config.mano_path
        )
        
        legacy_config = LegacyConfig(args)
        
        if self.config.use_gpu and torch.cuda.is_available():
            device_str = f"cuda:{self.config.device_id}"
        else:
            device_str = "cpu"
        
        self.device = torch.device(device_str)
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.reconstructor = HandReconstructor(legacy_config, device=self.device)
            self.logger.info("HandReconstructor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize HandReconstructor: {e}")
            raise

    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of frames or a full video.
        
        Args:
            input_data: Dictionary containing:
                - frames: List of image frames (numpy arrays)
                - fps: Video FPS (optional)
        
        Returns:
            Dictionary containing reconstruction results:
                - reconstruction: Dict with keys 'left', 'right', 'fov_x'
                - frames: Passed through
        """
        frames = input_data.get("frames")
        if not frames:
            raise ValueError("No frames provided for reconstruction")
        
        self.logger.info(f"Reconstructing hands for {len(frames)} frames...")
        
        # Identify if we need batching. The HandReconstructor might handle full video, 
        # but for memory safety we might want to be careful. 
        # However, the recon core seems to do temporal smoothing or optimization over the sequence.
        # So passing the full sequence is preferred if possible.
        
        # Note: HandReconstructor.recon takes a list of images
        try:
            recon_results = self.reconstructor.recon(frames)
            
            # Post-process or validation could happen here
            
            return {
                **input_data,  # Pass through existing data
                "reconstruction": recon_results
            }
            
        except Exception as e:
            self.logger.error(f"Reconstruction failed: {e}")
            raise

