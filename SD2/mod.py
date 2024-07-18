#!/bin/env python
import torch
from PIL import Image
from torchvision import transforms
from diff_pipe import StableDiffusionDiffImg2ImgPipeline

device = "cuda"

def new(pretrained):
	return StableDiffusionDiffImg2ImgPipeline.from_pretrained(pretrained, torch_dtype=torch.float16).to(device)

def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image * 2. - 1.
    image = image.unsqueeze(0).to(device)
    return image

def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map

def inpaint(pipeline):
	image = preprocess_image(Image.open("/run/user/1000/input.png"))
	map = 1. - preprocess_map(Image.open("/run/user/1000/mask.png")) * 7 / 9.

	prompt=['']
	#prompt=['night sky over a dark pine forest and meadow']
	prompt=['night sky over a meadow']
	negative_prompt=['text, watermark, signature, caption, people']
	#negative_prompt=["blurry, shadow polaroid photo, scary angry pose"]
	#guidance_scale=7,
	pipeline(prompt=prompt, image=image, num_images_per_prompt=1, negative_prompt=negative_prompt, map=map, num_inference_steps=20).images[0].save("/run/user/1000/output.png")
