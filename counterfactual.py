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


pretrained_path = './experiments/240602_3'
rank = 0
device = torch.device('cuda', rank)

seed = 2000 
batch_size = 4

save_dir = "./counterfactual_results/"
os.makedirs(save_dir, exist_ok=True)

# input_path = './avatars64x64/010尹哲晖-信科21.jpg'
# input_path = './avatars64x64/082姜峰 信科21.jpg'
# input_path = './avatars64x64/011房奕嘉-数院20.jpg'

# input_path = './avatars64x64/013薛珙乐-信科21.jpg'
# input_path = './avatars64x64/075康俊杰-信科22.jpg'
# input_path = './avatars64x64/073孙嘉旋 计算机21.jpg'
# input_path = './avatars64x64/057王湘源-信科21.jpg'

# input_path = './avatars64x64/050连祥-计算机20.jpg'
input_path = None

true_label = 0

if __name__ == '__main__':

    pipeline_pretrained = DDPMPipeline.from_pretrained(pretrained_path)
    pipeline_pretrained.to(device)
    pretrained_model = pipeline_pretrained.unet

    ddim_pipe = DDIMPipeline(unet=pretrained_model, scheduler=DDIMScheduler(num_train_timesteps=1000))

    # try ddpm
    ddim_pipe = pipeline_pretrained.to(device)


    if input_path is None:
        tmp_imgs = ddim_pipe.sample(batch_size=batch_size, generator=torch.Generator(device='cpu').manual_seed(seed), output_type='numpy', num_inference_steps=50).images
        tmp_img = tmp_imgs[0]
        # save tmp_img to save_dir
        # tmp_img = np.transpose(tmp_img, (1, 2, 0))
        tmp_img_to_save = tmp_img * 255
        tmp_img_to_save = cv2.cvtColor(tmp_img_to_save, cv2.COLOR_RGB2BGR)
        print(tmp_img_to_save.max(), tmp_img_to_save.min())
        cv2.imwrite(os.path.join(save_dir, f'true_{seed}.png'), tmp_img_to_save)
    else:
        tmp_img = cv2.imread(input_path)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        tmp_img = tmp_img / 255
        tmp_img = tmp_img.astype(np.float32)
        print(tmp_img.min(), tmp_img.max())

    ddim_pipe.to(device)


    tmp_img = (tmp_img * 2 - 1)*2-1
    tmp_img = torch.tensor(tmp_img, device=device).permute(2, 0, 1).reshape(1,3,64,64)
    # counterfactual
    noise = ddim_pipe.abduct(tmp_img, label_in = torch.tensor([true_label], device=device), num_inference_steps=20)

    # # try directly add noise
    # for t in range(1000):
    #     alpha_t = pipeline_pretrained.scheduler.alphas[t]
    #     beta_t = 1 - alpha_t
    #     tmp_img = alpha_t.sqrt() * tmp_img + beta_t * torch.randn_like(tmp_img)
    
    # noise = tmp_img * 3
    print(noise.max(), noise.min())

    noise_img = (noise/2+0.5).cpu().permute(0,2,3,1).numpy()
    # print(noise_img.max(), noise_img.min())
    noise_img = noise_img * 255
    noise_img = cv2.cvtColor(noise_img[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f'noise_{seed}.png'), noise_img)

    x_recon = ddim_pipe.reconstruct(noise, label_in = torch.tensor([true_label], device=device), output_type='numpy', num_inference_steps=20).images[0]
    x_cf = ddim_pipe.reconstruct(noise, label_in = torch.tensor([1-true_label], device=device), output_type='numpy', num_inference_steps=20).images[0]
    x_recon = x_recon * 255
    x_cf = x_cf * 255
    x_recon = cv2.cvtColor(x_recon, cv2.COLOR_RGB2BGR)
    x_cf = cv2.cvtColor(x_cf, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, f'recon_{seed}.png'), x_recon)
    cv2.imwrite(os.path.join(save_dir, f'cf_{seed}.png'), x_cf)

    # # random generate 
    # random_noise = torch.randn(1, 3, 64, 64, device=device)
    # print(random_noise.max(), random_noise.min())
    # x_cf_random_0 = ddim_pipe.reconstruct(random_noise, label_in = torch.tensor([0], device=device), output_type='numpy', num_inference_steps=50).images[0]
    # x_cf_random_0 = x_cf_random_0 * 255
    # x_cf_random_0 = cv2.cvtColor(x_cf_random_0, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(os.path.join(save_dir, f'cf_random0_{seed}.png'), x_cf_random_0)
    # x_cf_random_1 = ddim_pipe.reconstruct(random_noise, label_in = torch.tensor([1], device=device), output_type='numpy', num_inference_steps=50).images[0]
    # x_cf_random_1 = x_cf_random_1 * 255
    # x_cf_random_1 = cv2.cvtColor(x_cf_random_1, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(os.path.join(save_dir, f'cf_random1_{seed}.png'), x_cf_random_1)





