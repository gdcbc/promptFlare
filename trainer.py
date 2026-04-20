import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from torchvision.utils import save_image

from attention_control import CrossAttentionRecorder, FlareI2IAttnProcessor
from config import Config
from utils import (
    ensure_dir,
    list_images,
    load_image_tensor,
    pick_effective_timesteps,
    sample_timestep_positions,
    set_seed,
    soft_time_gate,
    total_variation,
)


class FlareI2ITrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        ensure_dir(cfg.output_dir)

        print("[*] Loading pipeline...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.infer_dtype,
            safety_checker=None if cfg.disable_safety_checker else None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        try:
            pipe.enable_attention_slicing()
            print("[*] attention slicing enabled.")
        except Exception:
            print("[*] attention slicing unavailable, skip.")

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[*] xformers enabled.")
        except Exception:
            print("[*] xformers not available, skip.")

        try:
            pipe.vae.enable_slicing()
            print("[*] VAE slicing enabled.")
        except Exception:
            pass

        try:
            pipe.vae.enable_tiling()
            print("[*] VAE tiling enabled.")
        except Exception:
            pass

        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.text_encoder.to(dtype=cfg.infer_dtype)
        self.vae.to(dtype=cfg.infer_dtype)
        self.unet.to(dtype=cfg.infer_dtype)

        try:
            self.unet.enable_gradient_checkpointing()
            print("[*] gradient checkpointing enabled.")
        except Exception:
            print("[*] gradient checkpointing unavailable, skip.")

        self.recorder = CrossAttentionRecorder()
        self._install_flare_processors()

        self.cached_quality_prompt_embeds = self.encode_prompt(cfg.quality_prompt)

        print("[*] Ready.")

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        with torch.no_grad():
            prompt_embeds = self.text_encoder(input_ids)[0]
        return prompt_embeds.to(dtype=self.cfg.infer_dtype)

    def _is_selected_processor(self, name: str) -> bool:
        if self.cfg.use_only_cross_attention and ("attn2" not in name):
            return False

        if not self.cfg.exclude_outermost:
            return True

        if "down_blocks.0" in name:
            return False
        if "up_blocks.3" in name:
            return False
        return True

    def _install_flare_processors(self):
        processors = {}
        selected, total = 0, 0

        for name in self.unet.attn_processors.keys():
            total += 1
            enabled = self._is_selected_processor(name)
            if enabled:
                selected += 1
            processors[name] = FlareI2IAttnProcessor(
                name=name,
                recorder=self.recorder,
                enabled=enabled,
            )

        self.unet.set_attn_processor(processors)
        print(f"[*] Installed custom attention processors: selected={selected}, total={total}")

    def encode_image_to_latent(self, image_01: torch.Tensor) -> torch.Tensor:
        image = image_01 * 2.0 - 1.0
        image = image.to(self.device, dtype=self.cfg.infer_dtype)

        posterior = self.vae.encode(image).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    def compute_ca_loss(self, protected_image: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        z0 = self.encode_image_to_latent(protected_image)

        effective_timesteps = pick_effective_timesteps(
            self.scheduler,
            num_inference_steps=cfg.num_inference_steps,
            strength=cfg.strength,
            device=self.device,
        )

        sampled = sample_timestep_positions(
            effective_timesteps,
            cfg.train_sample_timesteps,
        )

        if len(sampled) == 0:
            return torch.zeros((), device=self.device, dtype=torch.float32)

        prompt_embeds = self.cached_quality_prompt_embeds

        ca_loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        valid_count = 0
        base_noise = torch.randn_like(z0)
        total_steps = max(len(effective_timesteps), 1)

        for pos, timestep_int in sampled:
            timestep = torch.tensor([timestep_int], device=self.device, dtype=torch.long)

            zt = self.scheduler.add_noise(z0, base_noise, timestep)
            zt = self.scheduler.scale_model_input(zt, timestep)

            self.recorder.clear()
            self.recorder.set_mode("bos")
            with torch.no_grad():
                _ = self.unet(
                    zt,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=True,
                ).sample

            self.recorder.set_mode("full")
            _ = self.unet(
                zt,
                timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=True,
            ).sample

            layer_loss = self.recorder.mean_loss(self.device, torch.float32)

            norm_pos = 0.0 if total_steps <= 1 else (pos / float(total_steps - 1))
            w_t = soft_time_gate(
                norm_pos,
                cfg.tau_start,
                cfg.tau_end,
                cfg.time_gate_k,
            )

            ca_loss_total = ca_loss_total + layer_loss * float(w_t)
            valid_count += 1

            del zt, layer_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ca_loss_total / float(max(valid_count, 1))

    def train_one_image(self, image_path: str):
        cfg = self.cfg
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n[*] Training on {image_name}")

        clean = load_image_tensor(image_path, cfg.image_size).to(
            self.device,
            dtype=torch.float32,
        )
        delta = torch.zeros_like(clean, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=cfg.lr)

        image_out_dir = os.path.join(cfg.output_dir, image_name)
        ensure_dir(image_out_dir)

        best_loss = float("inf")
        best_delta = None

        for epoch in range(1, cfg.epochs + 1):
            optimizer.zero_grad(set_to_none=True)

            protected = (clean + delta).clamp(0.0, 1.0)

            loss_ca = self.compute_ca_loss(protected)
            loss_img = F.mse_loss(protected, clean)
            loss_tv = total_variation(delta)

            loss = (
                cfg.lambda_ca * loss_ca
                + cfg.lambda_img * loss_img
                + cfg.lambda_tv * loss_tv
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], cfg.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-cfg.eps, cfg.eps)
                delta.copy_((clean + delta).clamp(0.0, 1.0) - clean)

            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_delta = delta.detach().clone()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:04d} | "
                    f"loss={loss_value:.6f} | "
                    f"ca={float(loss_ca.item()):.6f} | "
                    f"img={float(loss_img.item()):.6f} | "
                    f"tv={float(loss_tv.item()):.6f}"
                )

            if epoch % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if best_delta is None:
            best_delta = delta.detach().clone()

        final_protected = (clean + best_delta).clamp(0, 1)

        save_image(final_protected, os.path.join(image_out_dir, "protected_final.png"))
        print(f"[*] Best loss for {image_name}: {best_loss:.6f}")
        print(f"[*] Saved: {os.path.join(image_out_dir, 'protected_final.png')}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_all(self):
        image_paths = list_images(self.cfg.image_dir, self.cfg.image_extensions)
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.cfg.image_dir}")

        print(f"[*] Found {len(image_paths)} images")

        for image_path in image_paths:
            try:
                self.train_one_image(image_path)
            except Exception as e:
                print(f"[!] Failed on {image_path}: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def main():
    cfg = Config()
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    trainer = FlareI2ITrainer(cfg)
    trainer.train_all()


if __name__ == "__main__":
    main()