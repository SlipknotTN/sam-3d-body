"""
Create a video from a folder of images.
"""

import argparse
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import glob

def do_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--image_suffix", type=str, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--output_video_path", type=str, required=True)
    return parser.parse_args()  

def main():
    args = do_parsing()
    print(args)

    input_paths = sorted(glob.glob(os.path.join(args.input_path, f"*{args.image_suffix}.jpg")))
    print(f"Found {len(input_paths)} images")

    os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)

    first_image = cv2.imread(input_paths[0])
    height, width, _ = first_image.shape

    video_writer = cv2.VideoWriter(
        args.output_video_path,
        apiPreference=cv2.CAP_FFMPEG,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=args.fps,
        frameSize=(width, height)
    )

    for input_path in tqdm(input_paths):
        image = cv2.imread(input_path)
        video_writer.write(image)

    video_writer.release()

    print(f"Video created successfully at {args.output_video_path}")

if __name__ == "__main__":
    main()
