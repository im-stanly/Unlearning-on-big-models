import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from diffusers.training_utils import compute_density_for_timestep_sampling

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_ID = "black-forest-labs/FLUX.1-dev"
IMAGE_DIR = "../flux_lora_images"
PROMPT = "a photo of TOK style"
OUTPUT_DIR = "../models_cache/flux-lora"

IMAGE_SIZE = 1024
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-4
RANK = 16

# Flux VAE Normalization
VAE_SCALE_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def prepare_img_ids(batch_size, height, width):
    """Generates the rotary positional IDs."""
    h_ids = torch.arange(height // 2)
    w_ids = torch.arange(width // 2)
    grid_h, grid_w = torch.meshgrid(h_ids, w_ids, indexing="ij")
    img_ids = torch.stack([grid_h, grid_w], dim=-1).reshape(-1, 2)
    img_ids = torch.cat([img_ids, torch.zeros(img_ids.shape[0], 1)], dim=-1)
    # return img_ids.unsqueeze(0).repeat(batch_size, 1, 1)
    return img_ids

def pack_latents(latents, batch_size, channels, height, width):
    """Packs (B, C, H, W) -> (B, L, C_packed)"""
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    return latents

def enable_gc(transformer):
    if hasattr(transformer, "_set_gradient_checkpointing"):
        transformer._set_gradient_checkpointing(True)
    elif hasattr(transformer, "gradient_checkpointing"):
        transformer.gradient_checkpointing = True
    elif hasattr(transformer, "_gradient_checkpointing"):
        transformer._gradient_checkpointing = True
    else:
        print("⚠️ Gradient checkpointing not supported")


# ==============================================================================
# DATASET
# ==============================================================================
class ImageDataset(Dataset):
    def __init__(self, image_dir, size=1024):
        self.paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(image)
        except Exception:
            return torch.zeros(3, 1024, 1024)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # 1. Setup Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    weight_dtype = torch.bfloat16

    # 2. Load Pipeline
    # This automatically loads CLIP, T5, VAE, Scheduler, and Transformer
    print("Loading Flux Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=weight_dtype
    ).to(device)

    # 3. Freeze Components
    # We only want to train the Transformer, so freeze everything else
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)   # CLIP
    pipe.text_encoder_2.requires_grad_(False) # T5
    pipe.transformer.requires_grad_(False)    # Freeze base transformer weights

    # 4. Attach LoRA to Transformer
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.train()
    
    # Optional: Enable gradient checkpointing for memory savings
    enable_gc(pipe.transformer)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.transformer.parameters()), 
        lr=LR
    )

    # 6. Data & Accelerator Prepare
    dataset = ImageDataset(IMAGE_DIR, size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Note: We pass pipe.transformer to prepare, not the whole pipe
    pipe.transformer, optimizer, dataloader = accelerator.prepare(
        pipe.transformer, optimizer, dataloader
    )

    # 7. Pre-compute Text Embeddings (Using the pipeline's encoders)
    print("Encoding prompt...")
    with torch.no_grad():
        # Encode with CLIP (text_encoder)
        clip_input = pipe.tokenizer(
            PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        pooled_projections = pipe.text_encoder(clip_input).pooler_output

        # Encode with T5 (text_encoder_2)
        t5_input = pipe.tokenizer_2(
            PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        encoder_hidden_states = pipe.text_encoder_2(t5_input)[0]

    # 8. Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")
        
        for step, pixel_values in enumerate(progress_bar):
            with accelerator.accumulate(pipe.transformer):
                
                # A. Encode Image (VAE)
                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                    latents = (latents - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR

                # B. Prepare Inputs
                bs, ch, h, w = latents.shape
                packed_latents = pack_latents(latents, bs, ch, h, w).to(dtype=weight_dtype)
                packed_latents.requires_grad_(True)
                img_ids = prepare_img_ids(bs, h, w).to(device=device, dtype=weight_dtype)
                txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3, device=device, dtype=weight_dtype)
                # txt_ids = torch.zeros(bs, encoder_hidden_states.shape[1], 3, device=device, dtype=weight_dtype)

                # C. Noise & Timesteps
                noise = torch.randn_like(packed_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bs, logit_mean=0.0, logit_std=1.0
                )
                timesteps = u.to(device) 

                # D. Add Noise
                sigmas = timesteps.view(-1, 1, 1)
                noisy_latents = (1 - sigmas) * packed_latents + sigmas * noise
                
                # E. Forward Pass
                print(f"DEBUG: Step {step} - Entering Transformer", flush=True)
                target = noise - packed_latents
                guidance_vec = torch.full((bs,), 1.0, device=device, dtype=weight_dtype)

                # Use pipe.transformer directly
                model_pred = pipe.transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    timestep=timesteps,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance_vec,
                    return_dict=False,
                )[0]
                print(f"DEBUG: Step {step} - Forward Done", flush=True)

                # F. Loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                print(f"DEBUG: Step {step} - Backward Start", flush=True)
                accelerator.backward(loss)
                print(f"DEBUG: Step {step} - Backward Done", flush=True)
                optimizer.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix(loss=loss.item())
                print(f"Step {step} complete. Loss: {loss.item()}", flush=True)

    # 9. Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipe.transformer.save_pretrained(OUTPUT_DIR)
        print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
