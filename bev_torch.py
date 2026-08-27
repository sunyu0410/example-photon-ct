import torch
import torch.nn.functional as F
import numpy as np
import cv2
from geometry import get_distance_slice_pt


def make_grid_5d(B, isocentre_idx, z_scales, ref_img):
    # B = mlc_masks_2d.size(0)                # 180 Control Points
    nz, ny, nx = ref_img.GetSize()
    D, H, W = ny, nx, nz  # Target 3D volume resolution in BEV space

    y_coords = torch.linspace(-1.0, 1.0, H)
    x_coords = torch.linspace(-1.0, 1.0, W)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    isocentre_grid = 2 * torch.tensor(isocentre_idx[::-1]) / torch.tensor([nx, ny, nz]) - 1
    x_c, y_c = isocentre_grid[2], isocentre_grid[0]  # Centred at the BEV origin

    X_transformed = (X - x_c) / z_scales + x_c
    Y_transformed = (Y - y_c) / z_scales + y_c
    Z_transformed = torch.zeros_like(
        X_transformed
    )  # Constant 0 because input depth is 1

    shared_grid_3d = torch.stack(
        [X_transformed, Y_transformed, Z_transformed], dim=-1
    )  # (90, 128, 128, 3)
    grid_5d = (
        shared_grid_3d.unsqueeze(0).float().expand(B, -1, -1, -1, -1)
    )  # [180, 90, 128, 128, 3]

    return grid_5d


def get_bev_torch(mlc_masks_2d, grid_5d):
    input_tensor = mlc_masks_2d.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, H, W)
    device = grid_5d.device

    beam_masks_3d = F.grid_sample(
        input_tensor.to(device),  # [B, 1, 1, nx, nz]
        grid_5d,  # [B, ny, nx, nz, 3]
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )

    # Output shape: (B, nx, ny, nz)
    final_bev_volumes = (
        beam_masks_3d.cpu().swapaxes(2, 3).squeeze(1).to(torch.uint8)
    )  # [B, nx, ny, nz]

    return final_bev_volumes


def cal_scales(ref_img, isocentre, src_mm):
    """Calcalte the ratios relative to distance(iso_slice, source)"""
    isocentre_idx = ref_img.TransformPhysicalPointToIndex(isocentre)

    nz, ny, nx = ref_img.GetSize()
    dist_iso = get_distance_slice_pt(ref_img, isocentre_idx[1], src_mm)
    dist_first = get_distance_slice_pt(ref_img, 0, src_mm)
    dist_last = get_distance_slice_pt(ref_img, ny - 1, src_mm)

    scales = np.linspace(dist_first, dist_last, ny) / dist_iso

    return scales


def mm2idx(ref_img, pts):
    return [ref_img.TransformPhysicalPointToIndex(i) for i in pts]


def draw_iso_mlc(ref_img, mlc):
    x, y, z = ref_img.GetSize()
    arr = np.zeros((z, x), np.uint8)
    shape_idx = [np.array(mm2idx(ref_img, pts)) for pts in mlc]

    cv2_pts = [
        np.array(idx[:, [0, 2]], dtype=np.int32).reshape((-1, 1, 2))
        for idx in shape_idx
    ]

    cv2_bev = cv2.fillPoly(arr, cv2_pts, color=1)
    return cv2_bev
