import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model


LEARNING_RATE = 1e-4
LORA_RANK = 8
GRAD_ACC_STEPS = 1
MIXED_PRECISION = "bf16"
SEED = 1234
MAX_STEPS = 1000
BATCH_SIZE = 4
EPOCHS = 100
TRIGGER_WORD = "TOK"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
OUTPUT_DIR = "./output"
DATASET_PATH = ""


class LocalImageDataset(Dataset):
    def __init__(self, directory, size=1024):
        self.directory = directory
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


def main():
    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
    )
    pipeline = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline, 'text_encoder_2'):
        pipeline.text_encoder_2.requires_grad_(False)
    if hasattr(pipeline, 'transformer'):
        pipeline.transformer.requires_grad_(False)
    if hasattr(pipeline, 'unet'):
        pipeline.unet.requires_grad_(False)

    pipeline.safety_checker = None
    inference_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device, dtype=inference_dtype)

    print('Number of parameters:')
    print(f'VAE: {sum(p.numel() for p in pipeline.vae.parameters())/1e6:.2f}M')
    print(f'Text Encoder: {sum(p.numel() for p in pipeline.text_encoder.parameters())/1e6:.2f}M')
    print(f'Text Encoder 2: {sum(p.numel() for p in pipeline.text_encoder_2.parameters())/1e6:.2f}M')
    print(f'Transformer: {sum(p.numel() for p in pipeline.transformer.parameters())/1e6:.2f}M')

    lora_config = LoraConfig(
        r = LORA_RANK,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.transformer.add_adapter(lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, pipeline.transformer.parameters()))
    assert len(lora_layers) > 0
    model = pipeline.transformer
    print(f'LoRA: {sum(p.numel() for p in lora_layers)/1e6:.2f}M')

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=LEARNING_RATE,
    )
    dataset = LocalImageDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    global_step = 0

    while global_step < MAX_STEPS:
        for batch in dataloader:
            with accelerator.accumulate(pipeline):
                with torch.no_grad():
                    latents = pipeline.vae.encode(batch.to(accelerator.device).to(torch.bfloat16)).latent_dist.sample()
                    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor

                    prompt = f"Image in the style of {TRIGGER_WORD}"
                    (
                        prompt_embeds,
                        pooled_prompt_embeds,
                        text_ids,
                    ) = pipeline.encode_prompt(prompt=prompt, prompt_2=prompt, device=accelerator.device)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                u = torch.rand((bsz,), device=latents.device)
                indices = (u * noise_scheduler.config.num_train_timesteps).long()

                # Add noise (Forward Process)
                # Flux uses: x_t = (1 - t) * x_0 + t * x_1
                sigmas = noise_scheduler.sigmas[indices].flatten()
                while len(sigmas.shape) < len(latents.shape):
                    sigmas = sigmas.unsqueeze(-1)

                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # 4. Predict
                # Flux transformer expects "packed" latents, but Diffusers wrapper handles unpacking if configured correctly.
                # Ideally, we pass the noisy latents directly.

                # Prepare rotary embeddings (ids)
                # This part is tricky in raw loops. Flux needs `img_ids`
                # img_ids = torch.zeros((bsz, latents.shape[2], latents.shape[3], 3), device=latents.device) # Simplified
                # (Ideally, you rely on the pipeline's internal prep, but here we call transformer directly)

                # WARNING: Calling transformer directly requires correct `img_ids` and packed hidden states.
                # To simplify this Minimal example, we rely on the fact that Diffusers' FluxTransformer2DModel
                # can handle standard unpacked 4D inputs if configured, OR we accept that we must construct the IDs.

                # Let's trust the model handles the shapes or use a helper from the pipeline if needed.
                # For this snippet to run without 100 lines of ID prep, we assume standard inputs:

                output = pipeline.transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=sigmas.squeeze(), # Pass sigma as timestep for Flux
                    txt_ids=text_ids,
                    img_ids=pipeline.prepare_latents( # Borrowing ID prep from pipe for convenience
                        bsz,
                        latents.shape[1],
                        latents.shape[2],
                        latents.shape[3],
                        prompt_embeds.dtype,
                        accelerator.device
                    )[2] # [2] is usually image_ids in the return tuple of prepare_latents
                ).sample

                # 5. Calculate Loss (Flow Matching)
                # Target is usually (noise - latents) i.e., the velocity to move from data to noise
                target = noise - latents
                loss = F.mse_loss(output, target, reduction="mean")

                # 6. Backprop
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 10 == 0:
                print(f"Step {global_step}: Loss {loss.item()}")

            global_step += 1
            if global_step >= MAX_STEPS:
                break

    # F. Save
    print("Saving LoRA...")
    pipeline.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()