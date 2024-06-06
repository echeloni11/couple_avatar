import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import load_dataset

from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, DDIMPipeline
from diffusers.utils import make_image_grid
import os

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path

# pretrained_path = './experiments/240602_3'
pretrained_path = 'harrym111/animeface_gender' # this is the huggingface path
rank = 0
device = torch.device('cuda', rank)


save_dir = "./cf_experiment_results_ddpm/"
os.makedirs(save_dir, exist_ok=True)

def counterfactual(file_name, true_label, ddim_pipe, inference_step=50):
    
    input_path = f"./avatars64x64/{file_name}"
    tmp_img = cv2.imread(input_path)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
    tmp_img = tmp_img / 255
    tmp_img = tmp_img.astype(np.float32)

    # for channel in range(tmp_img.shape[2]):
    #     mean = np.mean(tmp_img[:, :, channel])
    #     std = np.std(tmp_img[:, :, channel])
    #     tmp_img[:, :, channel] = (tmp_img[:, :, channel] - mean) / std

    mean = np.mean(tmp_img)
    std = np.std(tmp_img)
    tmp_img = (tmp_img - mean)/std

    tmp_img = torch.tensor(tmp_img, device=device).permute(2, 0, 1).reshape(1,3,64,64)
    # counterfactual
    noise = ddim_pipe.abduct(tmp_img, label_in = torch.tensor([true_label], device=device), num_inference_steps=inference_step)

    noise_img = (noise/2+0.5).cpu().permute(0,2,3,1).numpy()
    # print(noise_img.max(), noise_img.min())
    noise_img = noise_img * 255
    noise_img = cv2.cvtColor(noise_img[0], cv2.COLOR_RGB2BGR)
    save_img_dir = save_dir + file_name[0:3] + f"_{true_label}to{1-true_label}/"
    os.makedirs(save_img_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_img_dir, f'noise_{file_name[0:3]}.jpg'), noise_img)

    x_recon = ddim_pipe.reconstruct(noise, label_in = torch.tensor([true_label], device=device), output_type='numpy', num_inference_steps=inference_step).images[0]
    x_cf = ddim_pipe.reconstruct(noise, label_in = torch.tensor([1-true_label], device=device), output_type='numpy', num_inference_steps=inference_step).images[0]
    x_recon = x_recon * 255
    x_cf = x_cf * 255
    x_recon = cv2.cvtColor(x_recon, cv2.COLOR_RGB2BGR)
    x_cf = cv2.cvtColor(x_cf, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_img_dir, f'recon_{file_name[0:3]}.jpg'), x_recon)
    cv2.imwrite(os.path.join(save_img_dir, f'cf_{file_name[0:3]}.jpg'), x_cf)

def main():

    pipeline_pretrained = DDPMPipeline.from_pretrained(pretrained_path)
    pipeline_pretrained.to(device)
    pretrained_model = pipeline_pretrained.unet

    ddim_pipe = DDIMPipeline(unet=pretrained_model, scheduler=DDIMScheduler(num_train_timesteps=1000))
    ddim_pipe.to(device)

    # try ddpm
    ddim_pipe = pipeline_pretrained.to(device)

    for file_name in os.listdir('./avatars64x64'):
        print(f"Processing {file_name}")
        if file_name.endswith('.jpg'):
            counterfactual(file_name, 0, ddim_pipe, 50)
            counterfactual(file_name, 1, ddim_pipe, 50)


if __name__ == '__main__':
    main()