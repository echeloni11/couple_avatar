import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from dataclasses import dataclass
from datasets import load_dataset

from PIL import Image
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, DDIMPipeline
from diffusers.utils import make_image_grid
import os

from traintest import evaluate, train_loop, train_loop_no_accel



# training configuration
@dataclass
class TrainingConfig:
    date = "240602_3"
    image_size = 64  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    # config for classifier-free diffusion guidance
    drop_prob = 0.2  # drop probability for classifier-free diffusion
    guide_weight = 2
    #
    save_image_epochs = 5
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = f"experiments/{date}/"  # the model name locally and on the HF Hub

    pretrained_dir = None

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    # hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    # hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    num_class_embeds = 3  # 0 female, 1 for male, 2 for None (None when dropped)
    seed = 10
    rank = 2

def main():
    config = TrainingConfig()
    device = torch.device(f"cuda:{config.rank}" if torch.cuda.is_available() else "cpu")

    # # debug
    # device = "cpu"

    # prepare dataset
    dataset = load_dataset("imagefolder", data_dir="./images")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        labels = [int(label) for label in examples["label"]]
        return {"images": images, "labels": labels}

    dataset.set_transform(transform)

    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=config.train_batch_size, shuffle=True)

    # prepare model
    if config.pretrained_dir is not None:
        pipeline_pretrained = DDPMPipeline.from_pretrained(config.pretrained_dir)
        model = pipeline_pretrained.unet
        model.train()
    else:
        model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(192, 256, 384, 384, 512, 768, 1024),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            num_class_embeds=config.num_class_embeds,  # the dimension of the class token
        )

    model = model.to(device)

    sample_image = dataset['train'][0]["images"].unsqueeze(0)

    # prepare trainer
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # train
    train_loop_no_accel(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)


if __name__ == "__main__":
    main()

