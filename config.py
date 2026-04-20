from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class Config:
    model_id: str = r"H:\sd1.5\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    image_dir: str = r"H:\chengbaochun\promptflare\image_512"
    output_dir: str = r"H:\chengbaochun\promptflare\flare_i2i_output"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_dtype: torch.dtype = torch.float32
    infer_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    image_size: int = 512
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    epochs: int = 300
    lr: float = 2e-3
    grad_clip_norm: float = 1.0
    eps: float = 8.0 / 255.0
    save_every: int = 25

    num_inference_steps: int = 30
    strength: float = 0.6
    train_sample_timesteps: int = 3

    quality_prompt: str = "masterpiece, best quality, highly detailed"
    validate_prompts: List[str] = field(default_factory=lambda: [
        "a tiger standing in the forest",
        "a colorful dog in oil painting style",
        "turn the cat into a robot",
        "make it a Van Gogh painting",
    ])
    validate_cfg_scales: List[float] = field(default_factory=lambda: [7.5, 12.5])
    validate_strengths: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.8])

    lambda_ca: float = 1.0
    lambda_img: float = 0.15
    lambda_tv: float = 1e-5

    tau_start: float = 0.35
    tau_end: float = 0.75
    time_gate_k: float = 30.0

    use_only_cross_attention: bool = True
    exclude_outermost: bool = True
    disable_safety_checker: bool = True
