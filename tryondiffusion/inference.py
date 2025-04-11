# 📁 inference.py
import argparse
import os
import torch
from tryondiffusion.utils import load_model
from tryondiffusion.pipeline import TryOnPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_image_path", type=str, required=True)
    parser.add_argument("--cloth_image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model()
    pipeline = TryOnPipeline(model)

    output_path = os.path.join(args.output_dir, "tryon_result.png")
    pipeline.run(args.person_image_path, args.cloth_image_path, output_path)

if __name__ == "__main__":
    main()
