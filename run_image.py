import os
import argparse
import cv2
from pathlib import Path
import json
import torch
import matplotlib
import numpy as np
import pickle as pkl
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together
from tools.build_fov_estimator import run_moge_full

def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored

def do_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--bboxes",
        type=float,
        nargs="+",
        required=False,
        help="Bounding boxes in the format x1 y1 x2 y2, if not passed a ViT-Det will be used"
    )
    parser.add_argument("--intrinsics_json", type=str, required=False, help="Path to the intrinsic file")
    parser.add_argument("--normalize_depth_viz", action="store_true", default=False, help="Normalize the depth visualization")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = do_parsing()
    print(args)

    if args.bboxes is not None:
        assert len(args.bboxes) % 4 == 0, "Bboxes must be in the format x1 y1 x2 y2"
        bboxes = np.array(args.bboxes).reshape(-1, 4)
    else:
        bboxes = None
    assert os.path.exists(args.image_path), "Image path does not exist"

    cam_matrix = None
    if args.intrinsics_json is not None:
        intrinsic_params = json.load(open(args.intrinsics_json, "r"))
        if type(intrinsic_params) == dict:
            cam_matrix = torch.tensor(np.array(intrinsic_params["internal_highres"])).view(1, 3, 3)
            print(f"Cam matrix retrieved from {args.intrinsics_json}: {cam_matrix}")
            _dist_coeffs = np.array(intrinsic_params["fisheye_highres"])
        elif type(intrinsic_params) == list:
            cam_matrix = torch.tensor(np.array(intrinsic_params)).view(1, 3, 3)
            _dist_coeffs = None
        else:
            raise ValueError("Intrinsic parameters must be a dictionary or a list")

    # Set up the estimator
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    # Load and process image
    img = cv2.imread(args.image_path)
    outputs = estimator.process_one_image(
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        bboxes=bboxes,
        cam_int=cam_matrix
    )

    # WARNING: Not optimized, the model could be already run in the estimator.process_one_image function to get the intrinsics
    moge_data = run_moge_full(estimator.fov_estimator.fov_estimator, img, estimator.device)
    depth_viz = colorize_depth(
        depth=moge_data["depth"].cpu().numpy(),
        mask=moge_data["mask"].cpu().numpy(),
        normalize=args.normalize_depth_viz,
    )

    # Visualize and save results
    rend_img = visualize_sample_together(img, outputs, estimator.faces)
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, Path(args.image_path).stem + "_full.jpg"), rend_img.astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, Path(args.image_path).stem + "_depth.jpg"), depth_viz)

    pkl.dump(outputs, open(os.path.join(args.output_dir, Path(args.image_path).stem + "_sam3d_outputs.pkl"), "wb"))
    pkl.dump(moge_data, open(os.path.join(args.output_dir, Path(args.image_path).stem + "_moge_data.pkl"), "wb"))

    print("Done!")

if __name__ == "__main__":
    main()
