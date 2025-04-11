import argparse
from PIL import Image
from tryondiffusion.pipeline import TryOnPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--person_image_path", required=True)
parser.add_argument("--cloth_image_path", required=True)
parser.add_argument("--output_dir", default="results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load ảnh
person_img = Image.open(args.person_image_path).convert("RGB")
cloth_img = Image.open(args.cloth_image_path).convert("RGB")

# Khởi tạo pipeline và inference
pipeline = TryOnPipeline()
result = pipeline.infer(person_img, cloth_img)

# Lưu kết quả
output_path = os.path.join(args.output_dir, "tryon_result.png")
result.save(output_path)
print(f"✅ Saved result to {output_path}")
