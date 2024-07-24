#!/bin/env python
import torch
from PIL import Image
from torchvision import transforms
from diff_pipe import StableDiffusionDiffImg2ImgPipeline

device = "cuda"

#This is the default model, you can use other fine tuned models as well
pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                          torch_dtype=torch.float16).to(device)


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


with Image.open("MAT.png") as imageFile:
    image = preprocess_image(imageFile)

with Image.open("mask.png") as mapFile:
    map = 1. - preprocess_map(mapFile) * 5 / 9.

prompt=['night sky over a meadow']
negative_prompt=['text, watermark, signature, caption, people']
pipe(prompt=prompt, image=image, num_images_per_prompt=1, negative_prompt=negative_prompt, map=map, num_inference_steps=20).images[0].save("output.png")
