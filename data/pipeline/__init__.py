"""
VITRA Dataset Construction Pipeline

This package provides a complete pipeline for building Human Hand V-L-A datasets
from raw egocentric videos, following the three-stage approach described in the paper:

1. Hand and Camera Pose Reconstruction
2. Atomic Action Segmentation  
3. Language Annotation

Usage:
    from data.pipeline import DatasetBuilder
    
    builder = DatasetBuilder(config_path="configs/pipeline_config.yaml")
    builder.process_videos(video_dir="path/to/videos", output_dir="path/to/output")
"""

from .builder import DatasetBuilder
from .config import PipelineConfig

__all__ = ["DatasetBuilder", "PipelineConfig"]
