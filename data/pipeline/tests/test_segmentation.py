"""
Test script for VITRA Dataset Construction Pipeline.
Focuses on verifying the Speed Minima Segmentation algorithm with synthetic data.
"""

import unittest
import numpy as np
import os
import shutil
import logging
from pathlib import Path

# Adjust path to import the pipeline package
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.pipeline.config import SegmentationConfig, PipelineConfig
from data.pipeline.stages.action_segmentation import ActionSegmentationStage

# Configure logging to see pipeline output
logging.basicConfig(level=logging.INFO)

class TestActionSegmentation(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SegmentationConfig(
            fps=30.0,
            smoothing_sigma=1.0,
            minima_window_size=0.5, # 15 frames window
            min_segment_duration=0.2 # Short duration for testing
        )
        self.stage = ActionSegmentationStage(self.config)
        self.stage.initialize()

    def generate_synthetic_trajectory(self, length=100):
        """
        Generate a synthetic 1D trajectory (expanded to 3D) that has clear speed variations.
        Simulates 'move -> stop -> move' pattern.
        """
        # Create a time array
        t = np.linspace(0, 10, length)
        
        # Position: sin wave but with varying frequency to create speed changes
        # Speed ~ cos(t). Speed minima at peaks/valleys of position.
        x = np.sin(t) * 5 
        
        # Add some stops (constant position) to create zero speed segments
        # Let's manually construct a trajectory:
        # 0-30: Move from 0 to 1
        # 30-50: Stay at 1 (Speed ~ 0) -> Should be a split point / minima
        # 50-80: Move from 1 to 0
        # 80-100: Stay at 0
        
        traj = np.zeros((length, 3))
        
        # Segment 1: accelerating and decelerating move
        t1 = np.linspace(0, np.pi, 30)
        p1 = -np.cos(t1) * 0.5 + 0.5 # 0 to 1 curve
        
        # Segment 2: Stay
        p2 = np.ones(20)
        
        # Segment 3: Move back
        t3 = np.linspace(0, np.pi, 30)
        p3 = np.cos(t3) * 0.5 + 0.5 # 1 to 0 curve
        
        # Segment 4: Stay
        p4 = np.zeros(20)
        
        full_p = np.concatenate([p1, p2, p3, p4])
        
        # Assign to X axis, noisy Y and Z
        traj[:, 0] = full_p
        traj[:, 1] = np.random.normal(0, 0.001, length) # Minimal noise
        traj[:, 2] = np.random.normal(0, 0.001, length)
        
        return traj

    def test_trajectory_smoothing_and_velocity(self):
        """Test if trajectory smoothing and velocity calculation works."""
        traj = self.generate_synthetic_trajectory(100)
        
        # Access internal method for testing
        # Note: In Python unit tests for private methods, usually we test public interface
        # but here we want to verify the algorithm steps.
        
        # 1. Smoothing
        smoothed = self.stage.gaussian_filter1d(traj, sigma=1.0, axis=0)
        self.assertEqual(smoothed.shape, traj.shape)
        
        # 2. Velocity
        vel_vector = np.gradient(smoothed, axis=0)
        speed = np.linalg.norm(vel_vector, axis=1)
        
        # Expect low speed at index ~40 (middle of first stop)
        self.assertTrue(speed[40] < 0.01, f"Speed at stop should be low, got {speed[40]}")
        
        # Expect high speed at index ~15 (middle of move)
        self.assertTrue(speed[15] > 0.01, f"Speed at move should be high, got {speed[15]}")

    def test_segmentation_logic(self):
        """Test the full segmentation logic on synthetic data."""
        traj = self.generate_synthetic_trajectory(100)
        
        # Create input dictionary mocking previous stages
        # Convert trajectory to the dictionary format expected: {frame_idx: {'transl': ...}}
        hand_data = {}
        for i in range(len(traj)):
            hand_data[i] = {'transl': traj[i]}
            
        input_data = {
            "reconstruction": {
                "left": hand_data, # Use same traj for left
                "right": {}        # Empty right
            },
            "total_frames": 100,
            "fps": 30.0
        }
        
        result = self.stage.process(input_data)
        
        episodes = result["episodes"]
        left_eps = [e for e in episodes if e['anno_type'] == 'left']
        
        print("\nDetected Episodes (Left Hand):")
        for e in left_eps:
            print(f"  Frames: {e['start_frame']} -> {e['end_frame']} (Duration: {e['end_frame']-e['start_frame']})")
            
        # We expect segments around the moves.
        # Our trajectory has stops at ~30-50 and ~80-100.
        # Speed minima should be detected within these stops.
        
        self.assertTrue(len(left_eps) > 0, "Should detect at least one segment")
        
        # Check continuity
        for i in range(len(left_eps) - 1):
            curr_end = left_eps[i]['end_frame']
            next_start = left_eps[i+1]['start_frame']
            # They should be contiguous or close (depending on minima selection logic)
            self.assertEqual(curr_end, next_start, "Segments should be contiguous based on current logic")

    def test_missing_data_interpolation(self):
        """Test if the segmenter handles missing frames correctly."""
        traj = self.generate_synthetic_trajectory(50)
        
        hand_data = {}
        # Simulate missing frames by skipping indices
        for i in range(len(traj)):
            if 20 < i < 30: # Gap of 10 frames
                continue
            hand_data[i] = {'transl': traj[i]}
            
        input_data = {
            "reconstruction": {
                "left": hand_data,
                "right": {}
            },
            "total_frames": 50,
            "fps": 30.0
        }
        
        # Should not crash
        try:
            result = self.stage.process(input_data)
            left_eps = [e for e in result["episodes"] if e['anno_type'] == 'left']
            self.assertTrue(len(left_eps) > 0)
        except Exception as e:
            self.fail(f"Process crashed on missing data: {e}")

if __name__ == '__main__':
    unittest.main()
