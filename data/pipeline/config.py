"""
Configuration management for the dataset construction pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml
import os


@dataclass
class HandReconConfig:
    """Configuration for hand and camera pose reconstruction."""
    
    # Model paths
    hawor_model_path: str = "./weights/hawor/checkpoints/hawor.ckpt"
    detector_path: str = "./weights/hawor/external/detector.pt"
    moge_model_path: str = "Ruicheng/moge-2-vitl"
    mano_path: str = "./weights/mano"
    
    # Detection parameters
    detection_threshold: float = 0.2
    
    # Processing parameters
    batch_size: int = 100  # Number of frames to process at once
    use_gpu: bool = True
    device_id: int = 0


@dataclass
class SegmentationConfig:
    """Configuration for atomic action segmentation."""
    
    # 3D Wrist Velocity based Segmentation parameters
    
    # Trajectory smoothing
    # Gaussian filter sigma for smoothing the 3D trajectory
    smoothing_sigma: float = 1.0
    
    # Peak detection
    # Window size (in seconds) to look for local minima. 
    # Paper mentions 0.5s window. If fps is 30, this spans ~15 frames.
    minima_window_size: float = 0.5 
    
    # Filtering
    # Minimum duration for a valid segment (in seconds)
    min_segment_duration: float = 0.5
    # Maximum duration for a valid segment (in seconds)
    max_segment_duration: float = 10.0
    
    # Motion threshold
    # Minimum path length (in meters) to consider a segment valid
    min_motion_threshold: float = 0.01
    
    # Processing
    fps: float = 30.0  # Default FPS if not provided in input


@dataclass
class AnnotationConfig:
    """Configuration for language annotation."""
    
    # LLM settings
    llm_provider: str = "openai"  # "openai", "azure", "local"
    llm_model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Annotation parameters
    num_rephrase: int = 3  # Number of rephrased descriptions per action
    temperature: float = 0.7
    max_tokens: int = 256
    
    # Prompt templates
    action_prompt_template: str = """Based on the hand motion trajectory described below, provide a concise action description.

Hand: {hand_type}
Motion Summary:
- Start position: {start_pos}
- End position: {end_pos}
- Total displacement: {displacement:.3f} meters
- Duration: {duration} frames
- Primary direction: {direction}
- Finger motion: {finger_motion}

Describe this action in one sentence, focusing on what task the hand is performing.
Be specific and concise. Example formats:
- "Pick up the cup from the table."
- "Place the object on the shelf."
- "Open the drawer handle."

Action description:"""

    rephrase_prompt_template: str = """Rephrase the following action description {num_rephrase} different ways.
Keep the same meaning but vary the vocabulary and sentence structure.
Each rephrasing should be on a new line.

Original: {original_text}

Rephrasings:"""


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    
    # Supported video formats
    supported_formats: List[str] = field(default_factory=lambda: [".mp4", ".MP4", ".webm", ".avi"])
    
    # Frame extraction
    target_fps: Optional[float] = None  # None means use original fps
    resize_short_side: Optional[int] = None  # Resize for processing
    
    # Undistortion (for fisheye cameras)
    apply_undistortion: bool = False
    intrinsics_path: Optional[str] = None


@dataclass
class OutputConfig:
    """Configuration for output format."""
    
    # Output structure
    save_episodic_annotations: bool = True
    save_frame_index: bool = True
    save_visualization: bool = True
    
    # Compression
    compress_output: bool = True
    
    # Metadata format
    include_raw_trajectory: bool = True
    include_keypoints: bool = True


@dataclass
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    
    # Sub-configurations
    hand_recon: HandReconConfig = field(default_factory=HandReconConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # General settings
    num_workers: int = 4
    verbose: bool = True
    log_level: str = "INFO"
    
    # Paths
    cache_dir: str = "./cache"
    temp_dir: str = "./temp"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from a dictionary."""
        config = cls()
        
        if "hand_recon" in config_dict:
            config.hand_recon = HandReconConfig(**config_dict["hand_recon"])
        if "segmentation" in config_dict:
            config.segmentation = SegmentationConfig(**config_dict["segmentation"])
        if "annotation" in config_dict:
            config.annotation = AnnotationConfig(**config_dict["annotation"])
        if "video" in config_dict:
            config.video = VideoConfig(**config_dict["video"])
        if "output" in config_dict:
            config.output = OutputConfig(**config_dict["output"])
        
        # General settings
        for key in ["num_workers", "verbose", "log_level", "cache_dir", "temp_dir"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to a YAML file."""
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict
        return asdict(self)


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        return PipelineConfig.from_yaml(config_path)
    return get_default_config()
