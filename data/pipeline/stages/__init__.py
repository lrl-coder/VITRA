"""
Pipeline stages for dataset construction.

Each stage is a self-contained module that can be run independently or as part of the pipeline.
"""

from .hand_reconstruction import HandReconstructionStage
from .action_segmentation import ActionSegmentationStage
from .language_annotation import LanguageAnnotationStage
from .visualization import VisualizationStage
from .video_processor import VideoProcessor

# Optional: Camera pose estimation (requires DROID-SLAM)
try:
    from .camera_pose import CameraPoseEstimationStage
    __all__ = [
        "HandReconstructionStage",
        "ActionSegmentationStage", 
        "LanguageAnnotationStage",
        "VisualizationStage",
        "VideoProcessor",
        "CameraPoseEstimationStage",
    ]
except ImportError:
    __all__ = [
        "HandReconstructionStage",
        "ActionSegmentationStage", 
        "LanguageAnnotationStage",
        "VisualizationStage",
        "VideoProcessor",
    ]

