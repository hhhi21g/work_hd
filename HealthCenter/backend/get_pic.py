# sd_generate.py
import torch
from diffusers import StableDiffusionPipeline
import os

# 只加载一次模型（加速）
model_path = "D:\\AI\\checkpoints\\stable-diffusion-v1-5"  # 替换为你的路径
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

def generate_image(prompt, output_path):
    image = pipe(prompt).images[0]
    image.save(output_path)
    return output_path
