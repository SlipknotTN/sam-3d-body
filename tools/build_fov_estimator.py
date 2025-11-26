# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


class FOVEstimator:
    def __init__(self, name="moge2", device="cuda", **kwargs):
        self.device = device

        if name == "moge2":
            print("########### Using fov estimator: MoGe2...")
            self.fov_estimator = load_moge(device, **kwargs)
            self.fov_estimator_func = run_moge

            self.fov_estimator.eval()
        else:
            raise NotImplementedError

    def get_cam_intrinsics(self, img, **kwargs):
        return self.fov_estimator_func(self.fov_estimator, img, self.device, **kwargs)


def load_moge(device, path=""):
    from moge.model.v2 import MoGeModel

    if path == "":
        path = "Ruicheng/moge-2-vitl-normal"
    moge_model = MoGeModel.from_pretrained(path).to(device)
    return moge_model


def run_moge(model, input_image, device):
    # We expect the image to be RGB already
    H, W, _ = input_image.shape
    input_image = torch.tensor(
        input_image / 255, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    # Infer w/ MoGe2
    moge_data = model.infer(input_image)

    # get intrinsics
    intrinsics = denormalize_f(moge_data["intrinsics"].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]

    # override hfov with v_focal, forcing symmetrical fov
    intrinsics[0, 0] = v_focal
    # add batch dim
    cam_intrinsics = intrinsics[None]

    return cam_intrinsics

def run_moge_full(model, input_image, device):
    """
    `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
    The maps are in the same size as the input image. 
    {
        "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
        "depth": (H, W),        # depth map
        "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
        "mask": (H, W),         # a binary mask for valid pixels. 
        "intrinsics": (3, 3),   # normalized camera intrinsics
    }
    """
    input_image = torch.tensor(
        input_image / 255, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    # Infer w/ MoGe2
    moge_data = model.infer(input_image)

    return moge_data


def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    # s_norm = norm_K[0][1]   # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    # s_abs = s_norm * width
    s_abs = 0

    # Construct absolute K matrix
    abs_K = torch.tensor(
        [[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]]
    )
    return abs_K
