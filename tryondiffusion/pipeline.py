import torch
from torchvision import transforms
from PIL import Image
import os

class TryOnPipeline:
    def __init__(self, ckpt_path="checkpoints/viton512_v2.ckpt", device="cuda"):
        # Giả định có một model class tên là TryOnModel (chị đổi lại nếu tên khác)
        from tryondiffusion.model import TryOnModel

        self.device = device
        self.model = TryOnModel()
        state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def infer(self, person_image, cloth_image):
        # Resize + transform
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        person = transform(person_image).unsqueeze(0).to(self.device)
        cloth = transform(cloth_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(person, cloth)

        output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
        return output_image
