<div align="center">

<h1 align="center"><span
    style="font-family: 'Courier New', Courier, monospace; font-size: 115%;"><span style="font-size: 130%;">V</span>ITRA</span>:<br><span
    style="font-size:2.22rem;">利用现实人类活动视频进行机器人操纵的<br>可扩展视觉-语言-动作模型预训练
    </span></h1>

<p align="center">
    <a href="https://arxiv.org/abs/2510.21571"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
    <a href='https://microsoft.github.io/VITRA/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
    <a href='https://huggingface.co/VITRA-VLA/VITRA-VLA-3B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
    <a href='https://huggingface.co/datasets/VITRA-VLA/VITRA-1M'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow'></a>
    <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-orange' alt='License'></a>
</p>

<p align="center"><img src="assets/teaser.jpg" width="100%" alt="VITRA Teaser"></p>

<div align="justify">
<span style="font-family: 'Courier New', Courier, monospace; font-size: 115%;"><span style="font-size: 130%;">V</span>ITRA</span> 是一种利用大规模、无脚本、真实世界人类手部活动视频来预训练机器人操纵视觉-语言-动作 (VLA) 模型的新方法。我们将人手视为灵巧的机器人末端执行器，证明了没有任何标注的真实野外第一视角（egocentric）人类视频可以转换为在任务粒度和标签方面与现有机器人 V-L-A 训练数据完全对齐的数据格式。我们创建了一个包含超过 100 万个片段的人手 V-L-A 数据集。我们进一步开发了一个在该数据集上训练的、带有因果动作 Transformer 的 VLA 模型。它在全新场景中展现了强大的零样本（zero-shot）人手动作预测能力，并作为实物机器人操纵的少样本微调和适配的基石。
<br>
<br>

***有关视频演示，请参考我们的 [项目主页](https://microsoft.github.io/VITRA/)。***
</div>

<br>

</div>

---

## 🚩 新闻与更新
*   **[2025-12-05]** 🚀 发布使用单张图像进行零样本推理的代码。
*   **[2025-11-30]** 🚀 我们的代码、预训练模型和数据集现已开源。
*   **[2025-10-24]** 🚀 **VITRA** 论文在 arXiv 上发布。

---
## 🤗 预训练模型与数据集

我们的预训练模型和数据集可在 Hugging Face Hub 上获取：
<table>
  <thead>
    <tr>
      <th>Hugging Face 模型</th>
      <th>参数量</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/VITRA-VLA/VITRA-VLA-3B" target="_blank"><code>VITRA-VLA-3B</code><a></td>
      <td style="font-size: 0.92em;">3B</td>
      <td style="font-size: 0.92em;">基于人手数据预训练的基础 VLA 模型。</td>
    </tr>
  </tbody>
</table>

**注意：我们的基础 VLA 模型是从 [Paligemma2](https://huggingface.co/google/paligemma2-3b-mix-224) 微调而来的。如果您无法访问 [Paligemma2](https://huggingface.co/google/paligemma2-3b-mix-224)，请在 [官方网站](https://huggingface.co/google/paligemma2-3b-mix-224) 上申请权限。**

<table>
  <thead>
    <tr>
      <th rowspan="2">Hugging Face 数据集</th>
      <th colspan="2" style="text-align: center;">子数据集</th>
    </tr>
    <tr>
      <th>数据集名称</th>
      <th>片段数量 (Episodes)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><a href="https://huggingface.co/datasets/VITRA-VLA/VITRA-1M" target="_blank"><code>VITRA-1M</code></a></td>
      <td><code>ego4d_cooking_and_cleaning</code></td>
      <td>454,244</td>
    </tr>
    <tr>
      <td><code>ego4d_other</code></td>
      <td>494,439</td>
    </tr>
    <tr>
      <td><code>epic</code></td>
      <td>154,464</td>
    </tr>
    <tr>
      <td><code>egoexo4d</code></td>
      <td>67,053</td>
    </tr>
    <tr>
      <td><code>ssv2</code></td>
      <td>52,718</td>
    </tr>
    <tr>
      <td><strong>总计</strong></td>
      <td><strong>1,222,918</strong></td>
    </tr>
  </tbody>
</table>

**注意：详见 [`data/data.md`](data/data.md) 以获取有关我们数据集的详细信息。**

## 📑 目录
- [1. 安装](#1-安装)
  - [1.1 训练 / 推理要求](#11-训练--推理要求)
  - [1.2 可视化要求](#12-可视化要求)
- [2. 使用人手图像进行推理](#2-使用人手图像进行推理)
- [3. 使用自定义机器人数据集进行微调](#3-使用自定义机器人数据集进行微调)
  - [3.1 数据准备](#31-数据准备)
  - [3.2 实现自定义 RoboDatasetCore](#32-实现自定义-robodatasetcore)
  - [3.3 计算数据集统计信息](#33-计算数据集统计信息)
  - [3.4 修改配置](#34-修改配置)
  - [3.5 运行脚本](#35-运行脚本)
- [4. 实物部署](#4-实物部署)
- [5. 人手 VLA 数据集利用](#5-人手-vla-数据集利用)
- [6. 从头开始进行人类数据预训练](#6-从头开始进行人类数据预训练)
  - [6.1 数据集准备](#61-数据集准备)
  - [6.2 计算数据集统计信息](#62-计算数据集统计信息)
  - [6.3 修改配置](#63-修改配置)
  - [6.4 运行脚本](#64-运行脚本)
- [引用](#引用)


---

## 1. 安装
### 1.1 训练 / 推理要求
我们建议使用 `conda` 管理环境。需要 PyTorch >= 2.3.0 和 CUDA >= 12.1（较低版本也可能运行，但我们尚未测试）。如果环境仅用于训练，建议使用更高版本的 PyTorch 以获得更快的训练速度。

```bash
# 克隆仓库
git clone https://github.com/microsoft/VITRA.git
cd VITRA

# 创建环境
conda create -n vitra python=3.10 -y
conda activate vitra

# 安装依赖
pip install -e .
```

<details>
<summary>点击查看详细系统要求</summary>

*   **操作系统**: Linux (推荐 Ubuntu 20.04/22.04)
*   **Python**: 3.10+
*   **CUDA**: 11.8+
*   **GPU**: 推理至少需要 16GB 显存，训练推荐使用 A100/H100。
</details>

### 1.2 可视化要求
如果您想在推理后**可视化**结果、运行**数据集可视化**，或从单张图像进行零样本人类手部动作预测，请按照以下说明操作。

**安装子模块**

请克隆子模块以进行手部姿态估计。
```bash
git submodule update --init --recursive
```

**安装库**

请使用以下命令安装用于可视化的额外模块：

```bash
pip install -e .[visulization] --no-build-isolation
```

<details>
<summary>如果在安装 <a href="https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file">PyTorch3D</a> 时遇到问题，请点击这里 </summary>

*   如果您在安装 [PyTorch3D](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file) 时遇到问题，请按照 [PyTorch3D](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file) 仓库提供的安装说明进行操作，或者尝试使用以下命令单独安装：

    ```bash
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable#egg=pytorch3d
    ```
</details>


如果您的系统未安装 [FFmpeg](https://github.com/FFmpeg/FFmpeg)，请先安装它。
```bash
sudo apt install ffmpeg
```

**MANO 手部模型**

我们重建的手部标签基于 MANO 手部模型。**我们只需要右手模型。** 模型参数可以从 [官方网站](https://mano.is.tue.mpg.de/index.html) 下载，并组织成以下结构：
```
weights/
└── mano/
    ├── MANO_RIGHT.pkl
    └── mano_mean_params.npz
```
请下载 [HaWoR](https://github.com/ThunderVVV/HaWoR) 的模型权重用于手部姿态估计：

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./weights/hawor/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P ./weights/hawor/checkpoints/
```

---

## 2. 使用人手图像进行推理
您可以使用我们的预训练模型，根据指令直接从**第一视角人手图像（横屏）**进行零样本 3D 人手动作预测。要从预先捕获的图像预测人类动作，请运行 [`scripts/run_human_inference.sh`](scripts/run_human_inference.sh)。这是一个简单的示例：
```bash
python scripts/inference_human_prediction.py \
    --config VITRA-VLA/VITRA-VLA-3B \
    --image_path ./examples/0002.jpg \
    --sample_times 4 \
    --save_state_local \
    --use_right \
    --video_path ./example_human_inf.mp4 \
    --mano_path ./weights/mano \
    --instruction "Left hand: None. Right hand: Pick up the picture of Michael Jackson." \
```
所有示例图像都是在**V-L-A 数据集中从未出现的房间**内使用手机拍摄的。它们还包含了完全**未见过的概念**，例如名人的照片。

用户可以使用自己的设备捕获图像，并直接使用记录的图像测试模型。
> **注意：**  
> 为了获得最佳推理质量，建议捕获与**人头**高度相近的**横屏**视图图像，以匹配自然的第一视角观察点。极度异常或扭曲的手部姿态/位置可能会导致推理失败。


以下是预测人类动作的**最小化使用示例**。


```python
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

# 加载配置
configs = load_config('VITRA-VLA/VITRA-VLA-3B')

# 如果提供了路径则覆盖配置
pretrained_path = 'VITRA-VLA/VITRA-VLA-3B'
statistics_path = 'VITRA-VLA/VITRA-VLA-3B'
configs['model_load_path'] = pretrained_path
configs['statistics_path'] = statistics_path

# 加载模型和标准化器
model = load_model(configs).cuda()
model.eval()

normalizer = load_normalizer(configs)

image_path = "your_image.jpg"
image = Image.open(image_path)
image = resize_short_side_to_target(image, target=224)
fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]])      # 在此处输入您的相机 FOV [fov_x, fov_y]

image = np.array(image)

# 在此处输入您的提示语。以预测右手动作为例。
instruction = "Left hand: None. Right hand: Pick up the phone on the table."  

# 初始化状态
# 状态向量结构 (总维度: 122):
#   - state_left [51]:      左手状态向量
#       * [0:3]    transl:          相机空间中的平移 (x, y, z，单位为米)
#       * [3:6]    global_orient:   以欧拉角表示的全局旋转 (xyz，单位为弧度)
#       * [6:51]   hand_pose:       45 个关节角度欧拉角 (15 个关节 × 3 轴，单位为弧度)
#   - beta_left [10]:       左手 MANO 形状参数
#   - state_right [51]:     右手状态向量 (结构同 state_left)
#       * [0:3]    transl:          相机空间中的平移 (x, y, z，单位为米)
#       * [3:6]    global_orient:   以欧拉角表示的全局旋转 (xyz，单位为弧度)
#       * [6:51]   hand_pose:       45 个关节角度欧拉角 (15 个关节 × 3 轴，单位为弧度)
#   - beta_right [10]:      右手 MANO 形状参数
state = np.zeros((normalizer.state_mean.shape[0],))             # 在此处输入您的手部状态
# 仅以使用右手状态为例。
# state_mask[0] 指示是否使用左手状态， 
# state_mask[1] 指示是否使用右手状态。
state_mask = np.array([False, True], dtype=bool)                # 在此处输入您的手部状态掩码。 


# 在此处输入您的 action_mask。形状: (W, 2)，其中 W 是分块大小 (chunk_size)。 
# action_mask[:, 0] 指示是否预测左手动作， 
# action_mask[:, 1] 指示是否预测右手动作。 
# 示例中左手全为 False，右手全为 True。
action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))  


# 标准化状态
norm_state = normalizer.normalize_state(state)

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

# 模型推理
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
norm_action = norm_action[0, :,:102]
# 反标准化预测动作
unnorm_action = normalizer.unnormalize_action(norm_action)
print("预测动作:", unnorm_action)
```

---

## 3. 使用自定义机器人数据集进行微调


我们的 VITRA 模型可作为特定机器人微调的起点（例如在 Xhand 或您的自定义机器人上）。

### 3.1 数据准备

我们使用**相机空间末端执行器 (EEF) 姿态**来代表人手的腕部姿态。发送给 **XHand 的遥操作命令**被视为灵巧手**关节动作**，检索到的 **XHand 关节角度**被用作手关节状态。在进行机器人数据微调之前，我们强烈建议将双手的 EEF **平移**和**旋转**坐标系与人手 V-L-A 数据集中使用的坐标系对齐。

将所有坐标对齐到**相机坐标系**：

- **X 轴**: 指向屏幕右侧（正方向）
- **Y 轴**: 指向屏幕下方（正方向）
- **Z 轴**: 指向远离相机的方向，垂直进入屏幕（正方向）

对于 **EEF 旋转原点**，请注意左右手之间的镜像关系。对齐旋转原点后，EEF 姿态应与下图中的情况匹配。左侧是人手 V-L-A 数据集的示例；右侧是 XHand 数据集对齐后的示例。

![coordinate_alignment](assets/coordinate_alignment.png)

此外，我们在 [`vitra/dataset/robot_dataset.py`](./vitra/dataset/robot_dataset.py) 中提供了 `transfer_xhand_to_human` 函数，由于该函数会将 XHand 关节角度映射到人手表示中最接近的自由度。（注意，这种 XHand 到人手的对齐**不需要离线预处理**。它将在 `RoboDatasetCore` 类的 `transform_trajectory` 函数内部进行轨迹转换时自动应用。）

如果您使用的是不同的灵巧手模型，我们建议实现类似的函数，将其关节配置与人手自由度对齐。


### 3.2 实现自定义 `RoboDatasetCore`

要使数据集加载器适配您自己的机器人数据，您应该在 [`vitra/dataset/robot_dataset.py`](./vitra/dataset/robot_dataset.py) 中创建 `RoboDatasetCore` 类的自定义实现。

通常需要重写以下方法以匹配**人类数据预训练格式**：

```python
def __init__(...):
    # 数据集初始化
def __len__(self):
    # 返回帧数
    ...

def __getitem__(self, idx):
    """
    返回样本字典
    """
    ...
    return {
        "instruction": instruction,
        "image_list": image_list,
        "image_mask": image_mask,
        "action_list": action_list,
        "action_mask": action_mask,
        "current_state": current_state,
        "current_state_mask": current_state_mask,
        "fov": fov,
        "intrinsics": self.intrinsics,
    }
```

`__getitem__` 返回的字典包含代表机器人状态、动作和观测的多个字段。这里我们以 **XHand** 配置为例进行说明：

- **左手**: 6-DoF EEF 姿态 + 12 关节角度 = 18-DoF
- **右手**: 6-DoF EEF 姿态 + 12 关节角度 = 18-DoF
- **状态或动作的维度排序**: [`left_eef_trans`, `left_eef_euler_rotation`, `left_hand_joint`, `right_eef_trans`, `right_eef_euler_rotation`, `right_hand_joint`]

</br>

| 键 (Key)                | 类型                     | 形状 (Shape)      | 描述                                                                                               |
|------------------------|--------------------------|-------------------|-----------------------------------------------------------------------------------------------------------|
| **instruction**        | `str`                    | —                 | 任务的自然语言描述。<br> 例如：`"Left hand: None. Right hand: {Right hand prompt}"` |
| **image_list**         | `np.ndarray (uint8)`     | `(1, H, W, C)`    | RGB 图像序列。<br>• `C=3` (通道) <br>• `H, W` = 高度和宽度                                  |
| **image_mask**         | `np.ndarray (bool)`      | `(1,)`            | 指示哪些帧是有效的（`1=有效`, `0=填充`）。                                                 |
| **action_list**        | `np.ndarray (float32)`   | `(T, 36)`         | 机器人动作序列（动作分块/action chunking）。<br>• `T` = 动作分块的长度                                                         |
| **action_mask**        | `np.ndarray (bool)`      | `(T, 2)`          | 指示哪些时间步包含有效的左手或右手动作。`action_mask[:, 0]` 对应左手，`action_mask[:, 1]` 对应右手。                                                      |
| **current_state**      | `np.ndarray (float32)`   | `(36,)`           | 当前时间步的机器人状态。                                                                     |
| **current_state_mask** | `np.ndarray (float32)`   | `(2,)`            | 指示哪只手的状态是有效的。`current_state_mask[0]` 对应左手，`current_state_mask[1]` 对应右手。                                           |
| **fov**                | `np.ndarray (float32)`   | `(2,)`            | 相机视场角。`[fov_x, fov_y]`                                                                                    |
| **intrinsics**         | `np.ndarray (float32)`   | `(3, 3)`          | 相机内参矩阵。在训练或模型推理期间不使用。                                     |




### 3.3 计算数据集统计信息

在训练之前，我们需要计算机器人数据集的**数据集统计信息**，特别是**状态**和**动作**变量的均值和标准差。这些统计数据将用于使用高斯标准化来**标准化动作**。

### 3.4 修改配置

准备好数据集并计算统计信息后，您可以编辑 `vitra/configs/robot_finetune.json` 文件来更新路径和其他相关设置。`pretrain_path` 应替换为预训练模型的路径或包含它的文件夹，以便正确执行微调。

在 `scripts/run_robot_finetune.sh` 中，确保输入您的 Hugging Face token 和 WANDB API key，以便进行身份验证和日志记录。

### 3.5 运行脚本

配置完成后，您可以通过运行以下命令开始微调：

```bash
bash scripts/run_robot_finetune.sh
```

<!-- TODO: 提供示例代码 -->
---

## 4. 实物部署
以下是在 XHand 平台上运行 **VITRA** 推理的示例。

```python
import json
import torch
import numpy as np
from PIL import Image
from vitra.models import VITRA_Paligemma, load_model
from vitra.utils.data_utils import resize_short_side_to_target
from vitra.datasets.human_dataset import pad_state_human, pad_action
from vitra.utils.data_utils import load_normalizer
from vitra.datasets.dataset_utils import (
    ActionFeature,
    StateFeature,
)
from vitra.datasets.robot_dataset import (
    transfer_xhand_to_human,
    transfer_human_to_xhand,
    pad_state_robot, pad_action
)
# 加载配置
configs = json.load(open('configs/robot_finetune.json'))
pretrained_path = 'checkpoints/finetuned_model.pt'
statistics_path = 'statistics/RoboData_statistics.json'
configs['model_load_path'] = pretrained_path
configs['statistics_path'] = statistics_path

# 加载模型和标准化器
model = load_model(configs).cuda()
model.eval()

normalizer = load_normalizer(configs)

image_path = "your_image.jpg"
image = Image.open(image_path)
image = resize_short_side_to_target(image, target=224)
fov = torch.tensor([[np.deg2rad(60.0), np.deg2rad(60.0)]])      # 在此处输入您的相机 FOV [fov_x, fov_y]

image = np.array(image)

# 在此处输入您的提示语。以预测右手动作为例。
instruction = "Left hand: None. Right hand: Pour the contents of the bottle into the pot."  

# 初始化状态
# 状态向量结构 (总维度: 36):
#   - 左手 [0:18]:
#       * [0:3]    transl:          相机空间中的腕部平移 (x, y, z，单位为米)
#       * [3:6]    global_orient:   以欧拉角表示的腕部旋转 (xyz，单位为弧度)
#       * [6:18]   hand_pose:       XHand 关节角度 (12 个关节，单位为弧度)
#   - 右手 [18:36]:
#       * [18:21]  transl:          相机空间中的腕部平移 (x, y, z，单位为米)
#       * [21:24]  global_orient:   以欧拉角表示的腕部旋转 (xyz，单位为弧度)
#       * [24:36]  hand_pose:       XHand 关节角度 (12 个关节，单位为弧度)
state = np.zeros((normalizer.state_mean.shape[0],))             # 在此处输入您的手部状态
# 仅以使用右手状态为例。
# state_mask[0] 指示是否使用左手状态， 
# state_mask[1] 指示是否使用右手状态。
state_mask = np.array([False, True], dtype=bool)                # 在此处输入您的手部状态掩码。 


# 在此处输入您的 action_mask。形状: (W, 2)，其中 W 是 chunk_size。 
# action_mask[:, 0] 指示是否预测左手动作， 
# action_mask[:, 1] 指示是否预测右手动作。 
# 示例中左手全为 False，右手全为 True。
action_mask = np.tile(np.array([[False, True]], dtype=bool), (model.chunk_size, 1))  


# 标准化状态
norm_state = normalizer.normalize_state(state)

unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212

unified_state, unified_state_mask = pad_state_robot(
    state = norm_state,
    state_mask = state_mask,
    state_dim = normalizer.state_mean.shape[0],
    unified_state_dim = unified_state_dim,
)
_, unified_action_mask = pad_action(
    actions=None,
    action_mask=action_mask,
    action_dim=normalizer.action_mean.shape[0],
    unified_action_dim=unified_action_dim
)
human_state, human_state_mask, _, human_action_mask = transfer_xhand_to_human(
    unified_state, unified_state_mask,
    None, unified_action_mask
)
# 模型推理
norm_action = model.predict_action(
    image = image,
    instruction = instruction,
    current_state = human_state.unsqueeze(0),
    current_state_mask = human_state_mask.unsqueeze(0),
    action_mask_torch = human_action_mask.unsqueeze(0),
    num_ddim_steps = 10,
    cfg_scale = 5.0,
    fov = fov,
    sample_times = 1
)
norm_action = norm_action[0, :,:102]
norm_robot_action = transfer_human_to_xhand(norm_action)
# 反标准化预测动作
unnorm_action = normalizer.unnormalize_action(norm_robot_action)
print("预测动作形状:", unnorm_action.shape)
# 结果为 2 个 18-DoF 动作，共 16 步，形状为 [16, 36]
```

---

## 5. 人手 VLA 数据集利用


我们发布了 **Human Hand V-L-A** 数据集，它将“野外”视频转换为机器人对齐的 `(图像, 指令, 动作)` 元组。

您可以从 [VITRA-1M](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) 下载数据集标注。
下载后，请使用以下命令解压所有 `.gz` 文件：
```bash
tar -xzvf ego4d_cooking_and_cleaning.tar.gz
tar -xzvf ego4d_other.tar.gz
tar -xzvf egoexo4d.tar.gz
tar -xzvf ssv2.tar.gz
tar -xzvf epic.tar.gz
```
有关数据集、其结构和使用说明的详细信息，请参考 [`data/data.md`](data/data.md) 文件。


----


## 6. 从头开始进行人类数据预训练

要在 **Human-VLA** 数据集上重现我们的预训练结果：

### 6.1 数据集准备

首先，请按照 [`data/data.md`](data/data.md) 中的说明下载视频数据集和相应的标注文件。下载后，在进一步预处理之前对视频进行 **[去畸变 (undistortion)](data/data.md)**，以校正鱼眼和镜头畸变。

#### 关于 EgoExo4D 视频预处理的注意事项

去畸变后的 EgoExo4D 视频由于鱼眼校正而包含黑边。我们应用了额外的预处理步骤来移除这些黑边：首先将所有 EgoExo4D 帧大小调整为 448×448，然后中心裁剪为 256×256。内参的变化已在 `vitra/datasets/human_dataset.py` 中处理，其中相机内参的计算包含了裁剪变换并相应地调整了相机 FOV。


准备好后，按以下目录结构组织数据：
```
Data_root/
├── Video/
│   ├── Ego4D_root/
│   │   ├── {视频名称1}.mp4
│   │   ├── {视频名称2}.mp4
│   │   └── ...
│   ├── Epic-Kitchen_root/
│   │   ├── {视频名称1}.MP4
│   │   ├── {视频名称2}.MP4
│   │   └── ...
│   ├── EgoExo4D_root/
│   │   ├── {视频名称1}.mp4
│   │   ├── {视频名称2}.mp4
│   │   └── ...
│   └── Somethingsomething-v2_root/
│       ├── {视频名称1}.webm
│       ├── {视频名称2}.webm
│       └── ...
└── Annotation/
    ├── ego4d_cooking_and_cleaning/
    ├── ego4d_other/
    ├── egoexo4d/
    ├── epic/
    ├── ssv2/
    └── statistics/
```

#### 注意：

- `Video/` 包含来自不同来源（如 Ego4D、Epic-Kitchen、EgoExo4D 和 Something-Something v2）的所有原始视频文件。
- `Annotation/` 包含与相应视频对齐的标注文件。

#### (可选) 训练加速技巧

为了进一步加速训练，我们建议执行以下可选的预处理步骤：

1. 调整所有处理后的视频大小，使短边为 224，以实现更快的解码并减少内存占用（在调整大小时保持纵横比）。
2. 将长视频分割成每个最多 2000 帧的短片段。
   使用以下格式命名片段：`{video_name}_part{part_index}.mp4`，其中 `part_index` 从 **1** 开始。
3. 在 `configs/human_pretrain.json` 中设置：
```json
"clip_len": 2000
```
这显著提高了训练期间的数据加载速度，特别是对于大规模数据集。

### 6.2 计算数据集统计信息

您可以直接使用下载的数据集中提供的预计算统计信息。
或者，您可以通过运行以下命令来计算数据集统计信息（状态和动作的均值和标准差）。

```bash
python vitra/datasets/calculate_statistics.py --save_folder data_root/statistics
```

### 6.3 修改配置

编辑 `vitra/configs/human_pretrain.json` 文件以更新路径和其他相关设置。

在 `scripts/run_human_pretrain.sh` 中，确保输入您的 Hugging Face token 和 WANDB API key，以便进行身份验证和日志记录。

### 6.4 运行脚本

配置完成后，您可以通过运行以下命令开始预训练：
```bash
# 分布式训练
bash scripts/run_human_pretrain.sh
```


---

## 引用

如果您发现我们的工作在您的研究中有用，请引用：

```bibtex
@article{li2025vitra,
  title={Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos},
  author={Qixiu Li and Yu Deng and Yaobo Liang and Lin Luo and Lei Zhou and Chengtang Yao and Lingqi Zeng and Zhiyuan Feng and Huizhi Liang and Sicheng Xu and Yizhong Zhang and Xi Chen and Hao Chen and Lily Sun and Dong Chen and Jiaolong Yang and Baining Guo},
  journal={arXiv preprint arXiv:2510.21571},
  year={2025}
}
```
