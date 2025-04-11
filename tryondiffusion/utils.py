import os
import torch
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path, image_size=512):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def save_image(tensor, path):
    image = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(image)
    image.save(path)

def load_model(model_path="weights/model.pth", device="cuda"):
    from diffusers import StableDiffusionPipeline
    model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    return model
