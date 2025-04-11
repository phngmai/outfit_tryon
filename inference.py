import argparse
from tryondiffusion.pipeline import TryOnPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_image_path", type=str, required=True)
    parser.add_argument("--cloth_image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="results/tryon_result.png")
    args = parser.parse_args()

    pipeline = TryOnPipeline()
    pipeline(
        person_img_path=args.person_image_path,
        cloth_img_path=args.cloth_image_path,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
