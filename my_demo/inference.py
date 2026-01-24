# export HF_ENDPOINT=https://hf-mirror.com
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import numpy as np
from PIL import Image
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target, load_normalizer
from vitra.datasets.human_dataset import pad_state_human, pad_action
from vitra.utils.config_utils import load_config
from vitra.datasets.dataset_utils import (
    ActionFeature,
    StateFeature,
)

# Load configs
configs = load_config('VITRA-VLA/VITRA-VLA-3B')

# Override config if provided
pretrained_path = 'VITRA-VLA/VITRA-VLA-3B'
statistics_path = 'VITRA-VLA/VITRA-VLA-3B'
configs['model_load_path'] = pretrained_path
configs['statistics_path'] = statistics_path

# Load model and normalizer
model = load_model(configs).cuda()
model.eval()

normalizer = load_normalizer(configs)

image_path = "my_demo/image.png"
image = Image.open(image_path)
image = resize_short_side_to_target(image, target=224)
# 显式指定 dtype 为 float32，并移动到 cuda
fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]], dtype=torch.float32).cuda()

image = np.array(image)
print(f"[DEBUG] Image shape: {image.shape}")

# Input your prompt here. Only predict the right hand action as an example.
instruction = "Left hand: None. Right hand: Pick up the phone on the table."  

# Initialize state
# State vector structure (total dimension: 122):
#   - state_left [51]:      Left hand state vector
#       * [0:3]    transl:          Translation in camera space (x, y, z in meters)
#       * [3:6]    global_orient:   Global rotation as Euler angles (xyz, in radians)
#       * [6:51]   hand_pose:       45 joint angles as Euler angles (15 joints × 3 axes, in radians)
#   - beta_left [10]:       Left hand MANO shape parameters
#   - state_right [51]:     Right hand state vector (same structure as state_left)
#       * [0:3]    transl:          Translation in camera space (x, y, z in meters)
#       * [3:6]    global_orient:   Global rotation as Euler angles (xyz, in radians)
#       * [6:51]   hand_pose:       45 joint angles as Euler angles (15 joints × 3 axes, in radians)
#   - beta_right [10]:      Right hand MANO shape parameters
state = np.zeros((normalizer.state_mean.shape[0],))             # Input your hand state here
print(f"[DEBUG] Initial State shape: {state.shape}")

# Only use right hand state as an example.
# state_mask[0] indicates whether to use left hand state, 
# state_mask[1] indicates whether to use right hand state.
state_mask = np.array([False, True], dtype=bool)                # Input your hand state mask here. 
print(f"[DEBUG] State Mask shape: {state_mask.shape}")


# Input your action_mask here. Shape: (W, 2) where W is chunk_size. 
# action_mask[:, 0] indicates whether to predict left hand actions, 
# action_mask[:, 1] indicates whether to predict right hand actions. 
# All left hand False, all right hand True as an example.
action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))  
print(f"[DEBUG] Action Mask shape: {action_mask.shape}")


# Normalize state
norm_state = normalizer.normalize_state(state)
print(f"[DEBUG] Normalized State shape: {norm_state.shape}")


unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212

unified_state, unified_state_mask = pad_state_human(
    state = norm_state,
    state_mask = state_mask,
    action_dim = normalizer.action_mean.shape[0],
    state_dim = normalizer.state_mean.shape[0],
    unified_state_dim = unified_state_dim,
)
_, unified_action_mask = pad_action(
    actions=None,
    action_mask=action_mask,
    action_dim=normalizer.action_mean.shape[0],
    unified_action_dim=unified_action_dim
)

print(f"[DEBUG] Unified State shape: {unified_state.shape}")
print(f"[DEBUG] Unified State Mask shape: {unified_state_mask.shape}")
print(f"[DEBUG] Unified Action Mask shape: {unified_action_mask.shape}")


# Model inference
norm_action = model.predict_action(
    image = image,
    instruction = instruction,
    current_state = unified_state.unsqueeze(0),
    current_state_mask = unified_state_mask.unsqueeze(0),
    action_mask_torch = unified_action_mask.unsqueeze(0),
    num_ddim_steps = 10,
    cfg_scale = 5.0,
    fov = fov,
    sample_times = 1
)
print(f"[DEBUG] Raw Model Output (Norm Action) shape: {norm_action.shape}")

norm_action = norm_action[0, :,:102]
# Denormalize predicted action
unnorm_action = normalizer.unnormalize_action(norm_action)
print(f"[DEBUG] Final Unnormalized Action shape: {unnorm_action.shape}")
print("Action Shape:", unnorm_action.shape)
print("Predicted Action:", unnorm_action)