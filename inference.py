import os
import torch
from PIL import Image
from torchvision import transforms
from tryondiffusion.utils import load_model, save_image, preprocess_image
from tryondiffusion.pipeline import TryOnPipeline

def run_tryon(
    person_image_path: str,
    cloth_image_path: str,
    output_dir: str = "results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load và xử lý ảnh người
    print("[1/5] Loading person image...")
    person_img = preprocess_image(person_image_path)

    # 2. Load và xử lý ảnh quần áo
    print("[2/5] Loading cloth image...")
    cloth_img = preprocess_image(cloth_image_path)

    # 3. Tải mô hình TryOn
    print("[3/5] Loading TryOnDiffusion model...")
    model = load_model(device=device)

    # 4. Thực thi pipeline TryOn
    print("[4/5] Generating try-on result...")
    pipeline = TryOnPipeline(model=model, device=device)
    result_img = pipeline.run(person_img, cloth_img)

    # 5. Lưu kết quả
    result_path = os.path.join(output_dir, "tryon_result.png")
    print(f"[5/5] Saving result to {result_path}")
    save_image(result_img, result_path)

    return result_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI TryOn Inference")

    parser.add_argument("--person_image_path", type=str, required=True, help="Path to person image")
    parser.add_argument("--cloth_image_path", type=str, required=True, help="Path to cloth image")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()
    run_tryon(args.person_image_path, args.cloth_image_path, args.output_dir)
