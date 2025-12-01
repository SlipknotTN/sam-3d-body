import os
import argparse
import cv2
from notebook.utils import setup_sam_3d_body
import time
import json
import torch
import numpy as np
from tqdm import tqdm

def do_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--bboxes",
        type=float,
        nargs="+",
        required=False,
        help="Bounding boxes in the format x1 y1 x2 y2, if not passed a ViT-Det will be used"
    )
    parser.add_argument("--intrinsics_json", type=str, required=False, help="Path to the intrinsic file")
    parser.add_argument("--image_largest_side", type=int, required=False, default=None, help="Large side of the image")
    parser.add_argument("--warmup_runs", type=int, required=False, default=10, help="Number of warmup runs")
    parser.add_argument("--n_runs", type=int, required=False, default=100, help="Number of runs to profile")
    return parser.parse_args()

def main():
    args = do_parsing()
    print(args)

    assert os.path.exists(args.image_path), f"Input path {args.image_path} does not exist"

    if args.bboxes is not None:
        assert len(args.bboxes) % 4 == 0, "Bboxes must be in the format x1 y1 x2 y2"
        bboxes = np.array(args.bboxes).reshape(-1, 4)
    else:
        bboxes = None

    cam_matrix = None
    if args.intrinsics_json is not None:
        intrinsic_params = json.load(open(args.intrinsics_json, "r"))
        if type(intrinsic_params) == dict:
            cam_matrix = torch.tensor(np.array(intrinsic_params["internal_highres"])).view(1, 3, 3)
            print(f"Cam matrix retrieved from {args.intrinsics_json}: {cam_matrix}")
        elif type(intrinsic_params) == list:
            cam_matrix = torch.tensor(np.array(intrinsic_params)).view(1, 3, 3)
        else:
            raise ValueError("Intrinsic parameters must be a dictionary or a list")

    # Set up the estimator
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    image = cv2.imread(args.image_path)
    H, W, _ = image.shape
    print(f"Image size W x H: {W}x{H}")
    if args.image_largest_side is not None:
        if W > H:
            image = cv2.resize(image, (args.image_largest_side, int(args.image_largest_side * H / W)))
        else:
            image = cv2.resize(image, (int(args.image_largest_side * W / H), args.image_largest_side))
        H, W, _ = image.shape
        print(f"Resized image size W x H: {W}x{H}")

    for _ in tqdm(range(args.warmup_runs), desc="warmup runs"):
        estimator.process_one_image(
            img=image,
            bboxes=bboxes,
            cam_int=cam_matrix
        )

    process_times = []
    for _ in tqdm(range(args.n_runs), desc="profiling runs"):
        before = time.time()
        estimator.process_one_image(
            img=image,
            profiling=True,
            bboxes=bboxes,
            cam_int=cam_matrix
        )
        after = time.time()
        process_times.append(after - before)

    print(f"E2E time over {args.n_runs} runs: avg {np.mean(process_times):.3f} s, dev std {np.std(process_times):.3f} s, min {np.min(process_times):.3f} s, max {np.max(process_times):.3f} s")
    for key, times in estimator.profiling_times.items():
        if len(times) > 0:
            print(f"{key} time over {args.n_runs} runs: avg {np.mean(times):.3f} s, dev std {np.std(times):.3f} s, min {np.min(times):.3f} s, max {np.max(times):.3f} s")
        else:
            print(f"{key} time over {args.n_runs} runs: not available")
    print("Done!")

if __name__ == "__main__":
    main()
