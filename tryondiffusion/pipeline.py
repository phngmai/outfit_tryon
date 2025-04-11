# 📁 tryondiffusion/pipeline.py
import torch
from .utils import preprocess_image, save_image

class TryOnPipeline:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def run(self, person_image_path, cloth_image_path, output_path="results/tryon_result.png"):
        person_tensor = preprocess_image(person_image_path).to(self.device)
        cloth_tensor = preprocess_image(cloth_image_path).to(self.device)

        prompt = "a photo of a person wearing a stylish outfit"
        image = self.model(prompt=prompt, image=person_tensor, strength=0.75).images[0]

        image.save(output_path)
        print(f"✅ Saved to {output_path}")
