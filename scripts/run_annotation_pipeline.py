"""
Script to run the full VITRA Dataset Construction Pipeline on real videos.

Usage:
    python scripts/run_pipeline.py --input_dir data/examples/videos --output_dir data/my_custom_dataset
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.pipeline.builder import DatasetBuilder
from data.pipeline.config import PipelineConfig

def setup_logging(verbose=True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="VITRA Dataset Construction Pipeline")
    parser.add_argument("--input_dir", type=str, default="data/examples/videos", 
                        help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="data/output_dataset", 
                        help="Directory to save processed dataset")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to custom config yaml file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed logs")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger("PipelineRunner")
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
        
    # Load configuration
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = PipelineConfig()
        logger.info("Using default configuration")
        
    # Update config from args
    config.hand_recon.use_gpu = (args.device == "cuda")
    
    # Initialize Builder
    try:
        builder = DatasetBuilder()
        # Override config
        builder.config = config
        # Re-initialize stages with new config
        builder._init_stages()
        
        logger.info(f"Starting pipeline processing...")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {args.output_dir}")
        
        builder.process_directory(str(input_path), args.output_dir)
        
        logger.info("Pipeline processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'builder' in locals():
            builder.pipleline_cleanup()

if __name__ == "__main__":
    main()
