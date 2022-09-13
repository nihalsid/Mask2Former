from pathlib import Path
from tqdm import tqdm
import argparse
import os


def run(folder_in, folder_out):
    files = list(Path(folder_in).iterdir())
    Path(folder_out).mkdir(exist_ok=True)
    for f in tqdm(files):
        os.system(f"python demo.py --output {str(folder_out)} --predictions {str(folder_out)} --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input {str(f)} --opts MODEL.WEIGHTS ../checkpoints/model_final_f07440.pkl")       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="path to input rgb folder"
    )
    parser.add_argument(
        "--output",
        help="path to input rgb folder"
    )
    args = parser.parse_args()
    run(args.input, args.output)

