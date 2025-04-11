import torch
from diffusers import StableDiffusionInpaintPipeline
from tryondiffusion.utils import preprocess_image, save_image

class TryOnPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

    def __call__(self, person_img_path, cloth_img_path, output_path="tryon_result.png"):
        # Load and preprocess images
        person_img = preprocess_image(person_img_path).to(self.device)
        cloth_img = preprocess_image(cloth_img_path).to(self.device)

        # For now, blend cloth and person directly (placeholder logic)
        # You can insert segmentation/mask logic here to improve realism
        blend = person_img * 0.6 + cloth_img * 0.4

        # Save result
        save_image(blend, output_path)
        print(f"✅ Result saved to {output_path}")
