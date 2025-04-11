import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from tryondiffusion.utils import preprocess_image, save_image
from tryondiffusion.models.networks import build_model  # giả định đã clone đủ repo


class TryOnPipeline:
    def __init__(self, model_ckpt="checkpoints/viton512_v2.ckpt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model on {self.device}...")
        self.model = build_model()
        self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def infer(self, person_img: Image.Image, cloth_img: Image.Image) -> Image.Image:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        p_img = transform(person_img).unsqueeze(0).to(self.device)
        c_img = transform(cloth_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(p_img, c_img)  # giả định model đầu vào là (person, cloth)

        out_tensor = output.squeeze(0).cpu()
        out_img = transforms.ToPILImage()(out_tensor.clamp(0, 1))
        return out_img
