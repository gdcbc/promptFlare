import os

import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image

from config import Config
from utils import ensure_dir


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    # ========= 这里改你要测试的图片名 =========
    image_name = "image_1"

    # ========= 这里改测试 prompt =========
    test_prompts = [
        "a tiger standing in the forest",
        "a colorful dog in oil painting style",
        "turn the cat into a robot",
        "make it a Van Gogh painting",
    ]

    # ========= 这里改测试参数 =========
    test_strengths = [0.4, 0.6, 0.8]
    test_cfg_scales = [7.5, 12.5]

    clean_image_path = os.path.join(cfg.image_dir, f"{image_name}.jpg")
    if not os.path.exists(clean_image_path):
        clean_image_path = os.path.join(cfg.image_dir, f"{image_name}.png")

    protected_image_path = os.path.join(cfg.output_dir, image_name, "protected_final.png")
    save_dir = os.path.join(cfg.output_dir, image_name, "test_results")
    ensure_dir(save_dir)

    if not os.path.exists(clean_image_path):
        raise FileNotFoundError(f"Clean image not found: {clean_image_path}")

    if not os.path.exists(protected_image_path):
        raise FileNotFoundError(f"Protected image not found: {protected_image_path}")

    print("[*] Loading pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.infer_dtype,
        safety_checker=None if cfg.disable_safety_checker else None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    try:
        pipe.enable_attention_slicing()
        print("[*] attention slicing enabled.")
    except Exception:
        pass

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[*] xformers enabled.")
    except Exception:
        print("[*] xformers not available, skip.")

    clean_pil = load_pil(clean_image_path)
    protected_pil = load_pil(protected_image_path)

    clean_pil.save(os.path.join(save_dir, f"{image_name}_clean.png"))
    protected_pil.save(os.path.join(save_dir, f"{image_name}_protected.png"))

    for strength in test_strengths:
        for cfg_scale in test_cfg_scales:
            for idx, prompt in enumerate(test_prompts):
                print(f"[*] prompt={prompt} | strength={strength} | cfg={cfg_scale}")

                gen_clean = torch.Generator(device=device).manual_seed(cfg.seed)
                gen_protected = torch.Generator(device=device).manual_seed(cfg.seed)

                out_clean = pipe(
                    prompt=prompt,
                    image=clean_pil,
                    strength=float(strength),
                    guidance_scale=float(cfg_scale),
                    num_inference_steps=cfg.num_inference_steps,
                    generator=gen_clean,
                ).images[0]

                out_protected = pipe(
                    prompt=prompt,
                    image=protected_pil,
                    strength=float(strength),
                    guidance_scale=float(cfg_scale),
                    num_inference_steps=cfg.num_inference_steps,
                    generator=gen_protected,
                ).images[0]

                tag = f"s{strength}_cfg{cfg_scale}_p{idx}"
                out_clean.save(os.path.join(save_dir, f"{tag}_clean_edit.png"))
                out_protected.save(os.path.join(save_dir, f"{tag}_protected_edit.png"))

    print(f"[*] Test results saved to: {save_dir}")


if __name__ == "__main__":
    main()