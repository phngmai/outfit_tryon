import os
from PIL import Image

def apply_fake_tryon(person_image_path, cloth_image_path, output_dir="results"):
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Mở ảnh người và ảnh quần áo
    person_img = Image.open(person_image_path).convert("RGBA")
    cloth_img = Image.open(cloth_image_path).convert("RGBA")

    # Resize ảnh quần áo cho vừa với người (giả định cùng kích thước)
    cloth_img = cloth_img.resize(person_img.size)

    # Giả lập "thử đồ" bằng cách blend 2 ảnh lại
    blended = Image.blend(person_img, cloth_img, alpha=0.5)

    # Lưu kết quả
    output_path = os.path.join(output_dir, "tryon_result.png")
    blended.save(output_path)

    print(f"✅ Saved result to {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--person_image_path", type=str, required=True)
    parser.add_argument("--cloth_image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    apply_fake_tryon(args.person_image_path, args.cloth_image_path, args.output_dir)
