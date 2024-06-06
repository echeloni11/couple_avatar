import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset

from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, DDIMPipeline
from diffusers.utils import make_image_grid
import os

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline.sample(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    device = model.device
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            labels = batch["labels"]
            clean_images = clean_images.to(device)
            labels = labels.to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                # loss = F.mse_loss(noise_pred, noise)

                # Classifier-free guidance
                # drop the label (set it to 2) with probability `config.drop_prob`
                mask = torch.rand(bs, device=clean_images.device) > config.drop_prob
                labels = labels * mask.long() + 2 * ~mask.long()
                noise_pred = model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

def train_loop_no_accel(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # # 将模型和数据移动到 CPU
    # device = torch.device('cpu')
    # model.to(device)

    device = model.device

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            labels = batch["labels"]
            clean_images = clean_images.to(device)
            labels = labels.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Classifier-free guidance
            # drop the label (set it to 2) with probability `config.drop_prob`
            mask = torch.rand(bs, device=clean_images.device) > config.drop_prob
            
            # labels = labels * mask.long() + 2 * ~mask.long()
            labels = labels * mask + 2 * ~mask
            
            # debug
            # print(f"noisy_images.shape: {noisy_images.shape}, timesteps.shape: {timesteps.shape}, labels.shape: {labels.shape}")
            noise_pred = model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
        

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
            evaluate(config, epoch, pipeline)
            # if config.push_to_hub:
            #     upload_folder(
            #         repo_id=repo_id,
            #         folder_path=config.output_dir,
            #         commit_message=f"Epoch {epoch}",
            #         ignore_patterns=["step_*", "epoch_*"],
            #     )
            # else:
            pipeline.save_pretrained(config.output_dir)