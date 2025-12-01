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
from tools.vis_utils import visualize_sample_2d3d_together
from tools.build_fov_estimator import run_moge_full
import glob
from tqdm import tqdm

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
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, required=True, choices=["image", "folder"])
    parser.add_argument("--max_images", type=int, required=False, default=None, help="Maximum number of images to process from the folder")
    parser.add_argument(
        "--bboxes",
        type=float,
        nargs="+",
        required=False,
        help="Bounding boxes in the format x1 y1 x2 y2, if not passed a ViT-Det will be used"
    )
    parser.add_argument("--intrinsics_json", type=str, required=False, help="Path to the intrinsic file")
    parser.add_argument("--normalize_depth_viz", action="store_true", default=False, help="Normalize the depth visualization")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory to save the results")
    parser.add_argument("--save_sam_3d_outputs", action="store_true", default=False, help="Save the SAM3D outputs dict as a pickle file")
    parser.add_argument("--save_moge_full_data", action="store_true", default=False, help="Save the MoGe data dict as a pickle file")
    parser.add_argument("--save_moge_depth_only", action="store_true", default=False, help="Save the MoGe depth only as a numpy array")
    return parser.parse_args()

def main():
    args = do_parsing()
    print(args)

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
            _dist_coeffs = np.array(intrinsic_params["fisheye_highres"])
        elif type(intrinsic_params) == list:
            cam_matrix = torch.tensor(np.array(intrinsic_params)).view(1, 3, 3)
            _dist_coeffs = None
        else:
            raise ValueError("Intrinsic parameters must be a dictionary or a list")

    # Set up the estimator
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    if args.input_type == "image":
        input_paths = [args.input_path]
    elif args.input_type == "folder":
        input_paths = sorted(glob.glob(os.path.join(args.input_path, "*.jpg")))
        print(f"Found {len(input_paths)} images in the folder")
        if args.max_images is not None:
            input_paths = input_paths[:args.max_images]
            print(f"Using only {len(input_paths)} images")
    else:
        raise ValueError("Invalid input type")

    assert len(input_paths) > 0, "No images found"
    os.makedirs(args.output_dir, exist_ok=True)

    H, W, _ = cv2.imread(input_paths[0]).shape
    print(f"Image size W x H: {W}x{H}")

    for input_path in tqdm(input_paths):
        assert os.path.exists(input_path), "Image path does not exist"
        # Load and process image
        img = cv2.imread(input_path)
        outputs = estimator.process_one_image(
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            bboxes=bboxes,
            cam_int=cam_matrix
        )

        # Visualize and save results

        # Draw over original image
        
        # WARNING: Not optimized, the model could be already run in the estimator.process_one_image function to get the intrinsics
        moge_data = run_moge_full(estimator.fov_estimator.fov_estimator, img, estimator.device)
        depth_viz = colorize_depth(
            depth=moge_data["depth"].cpu().numpy(),
            mask=moge_data["mask"].cpu().numpy(),
            normalize=args.normalize_depth_viz,
        )

        img_keypoints, img_mesh = visualize_sample_2d3d_together(img, outputs, estimator.faces)
        depth_filename = "depth" if args.normalize_depth_viz else "depth_unnorm"
        cv2.imwrite(os.path.join(args.output_dir, Path(input_path).stem + "_img_keypoints.jpg"), img_keypoints.astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, Path(input_path).stem + "_img_meshes.jpg"), img_mesh.astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, Path(input_path).stem + f"_{depth_filename}.jpg"), depth_viz)

        # Draw over the depth image
        img_depth_keypoints, img_depth_mesh = visualize_sample_2d3d_together(depth_viz, outputs, estimator.faces)
        cv2.imwrite(os.path.join(args.output_dir, Path(input_path).stem + f"_{depth_filename}_keypoints.jpg"), img_depth_keypoints.astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, Path(input_path).stem + f"_{depth_filename}_meshes.jpg"), img_depth_mesh.astype(np.uint8))

        # Save results
        if args.save_sam_3d_outputs:
            pkl.dump(outputs, open(os.path.join(args.output_dir, Path(input_path).stem + "_sam3d_outputs.pkl"), "wb"))
        if args.save_moge_full_data:
            pkl.dump(moge_data, open(os.path.join(args.output_dir, Path(input_path).stem + "_moge_data.pkl"), "wb"))
        if args.save_moge_depth_only:
            np.save(os.path.join(args.output_dir, Path(input_path).stem + f"_moge_depth.npy"), moge_data["depth"].cpu().numpy())
       
    print("Done!")

if __name__ == "__main__":
    main()
