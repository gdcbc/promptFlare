import os
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


TORCHVISION_TO_TENSOR = transforms.ToTensor()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def list_images(image_dir: str, exts: Tuple[str, ...]) -> List[str]:
    paths = []
    for name in os.listdir(image_dir):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in exts):
            paths.append(os.path.join(image_dir, name))
    return sorted(paths)



def load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    return TORCHVISION_TO_TENSOR(img).unsqueeze(0)



def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().clamp(0, 1)[0]
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)



def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw



def soft_time_gate(norm_pos: float, tau_start: float, tau_end: float, k: float) -> float:
    s1 = 1.0 / (1.0 + math.exp(-k * (norm_pos - tau_start)))
    s2 = 1.0 / (1.0 + math.exp(-k * (norm_pos - tau_end)))
    return max(s1 - s2, 0.0)



def pick_effective_timesteps(scheduler, num_inference_steps: int, strength: float, device: torch.device):
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    effective = timesteps[t_start:]
    return effective



def sample_timestep_positions(effective_timesteps: torch.Tensor, sample_num: int):
    total = len(effective_timesteps)
    if total == 0:
        return []
    sample_num = min(sample_num, total)
    idxs = torch.linspace(0, total - 1, steps=sample_num).long().tolist()
    idxs = sorted(set(idxs))
    return [(pos, int(effective_timesteps[pos].item())) for pos in idxs]
