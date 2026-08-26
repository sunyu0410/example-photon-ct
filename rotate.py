import torch
import torch.nn.functional as F
import SimpleITK as sitk


def make_grid(
    tensor_img,  # Shape: (D, H, W) or (B, 1, D, H, W)
    spacing,  # (x, y, z) spacing in mm
    origin,  # (x, y, z) origin in mm
    isocentre,  # (x, y, z) rotation center in mm
    degrees_list,  # list or tensor of angles in degrees
    axis="z",  # "x", "y", or "z"
    bg_value=-1024.0,
):
    """
    Rotates a 3D image using PyTorch matching SimpleITK's physical space logic.
    Inputs for spacing, origin, and isocentre must be in (X, Y, Z) order.
    """
    dtype = tensor_img.dtype
    device = tensor_img.device

    # 1. Standardise tensor dimensions to 5D format: (B, 1, D, H, W)
    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
    elif tensor_img.dim() == 4:
        tensor_img = tensor_img.unsqueeze(1)

    _, _, D, H, W = tensor_img.shape

    # PyTorch GridSample coordinates: X=Width, Y=Height, Z=Depth
    size_xyz = torch.tensor([W, H, D], dtype=dtype, device=device)
    spacing_xyz = torch.as_tensor(spacing, dtype=dtype, device=device)
    origin_xyz = torch.as_tensor(origin, dtype=dtype, device=device)
    iso_xyz = torch.as_tensor(isocentre, dtype=dtype, device=device)
    angles_deg = torch.as_tensor(degrees_list, dtype=dtype, device=device)

    num_batch = len(angles_deg)

    # 2. Protect against zero-division for dimensions of size 1
    denom = size_xyz - 1.0
    # denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    # 3. Calculate mapping factors between [-1, 1] and Millimeters
    scale_mm = (denom * spacing_xyz) / 2.0
    translation_mm = scale_mm + origin_xyz

    # 4. Matrix: Normalized [-1, 1] to Physical Millimeters
    M_norm_to_phys = (
        torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(num_batch, 1, 1)
    )
    M_norm_to_phys[:, 0, 0] = scale_mm[0]
    M_norm_to_phys[:, 1, 1] = scale_mm[1]
    M_norm_to_phys[:, 2, 2] = scale_mm[2]
    M_norm_to_phys[:, 0, 3] = translation_mm[0]
    M_norm_to_phys[:, 1, 3] = translation_mm[1]
    M_norm_to_phys[:, 2, 3] = translation_mm[2]

    # 5. Matrix: Physical Millimeters to Normalized [-1, 1]
    M_phys_to_norm = (
        torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(num_batch, 1, 1)
    )
    M_phys_to_norm[:, 0, 0] = 1.0 / scale_mm[0]
    M_phys_to_norm[:, 1, 1] = 1.0 / scale_mm[1]
    M_phys_to_norm[:, 2, 2] = 1.0 / scale_mm[2]
    M_phys_to_norm[:, 0, 3] = -translation_mm[0] / scale_mm[0]
    M_phys_to_norm[:, 1, 3] = -translation_mm[1] / scale_mm[1]
    M_phys_to_norm[:, 2, 3] = -translation_mm[2] / scale_mm[2]

    # 6. Matrix: Inverse Physical Rotation
    # We use positive angles because grid_sample pulls from the source image
    rad = torch.deg2rad(-angles_deg)
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)

    M_rotate_inv = (
        torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(num_batch, 1, 1)
    )

    Cx, Cy, Cz = iso_xyz[0], iso_xyz[1], iso_xyz[2]

    assert axis == "z"
    M_rotate_inv[:, 0, 0] = cos_a
    M_rotate_inv[:, 0, 1] = -sin_a
    M_rotate_inv[:, 1, 0] = sin_a
    M_rotate_inv[:, 1, 1] = cos_a
    M_rotate_inv[:, 0, 3] = Cx - (cos_a * Cx - sin_a * Cy)
    M_rotate_inv[:, 1, 3] = Cy - (sin_a * Cx + cos_a * Cy)

    # 7. Compose the Final Transformation Matrix
    # Order: [Phys -> Norm] * [Rotate] * [Norm -> Phys]
    M_final = torch.bmm(M_phys_to_norm, torch.bmm(M_rotate_inv, M_norm_to_phys))

    # Extract the 3x4 affine matrix required by affine_grid
    batch_matrices = M_final[:, 0:3, 0:4]

    # 9. Resample
    grid = F.affine_grid(
        batch_matrices, size=(num_batch, 1, D, H, W), align_corners=True
    )

    return grid


def apply_grid(tensor_img, grid, bg_value=-1024.0):
    num_batch = grid.size(0)

    if tensor_img.dim() == 3:
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
    elif tensor_img.dim() == 4:
        tensor_img = tensor_img.unsqueeze(1)

    if tensor_img.size(0) != num_batch:
        batched_img = tensor_img.expand(num_batch, -1, -1, -1, -1) - bg_value
    else:
        batched_img = tensor_img - bg_value

    rotated_tensor = F.grid_sample(
        batched_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    rotated_tensor = rotated_tensor + bg_value

    return rotated_tensor.squeeze(1)


def rotate_image_new(
    tensor_img,  # Shape: (D, H, W) or (B, 1, D, H, W)
    spacing,  # (x, y, z) spacing in mm
    origin,  # (x, y, z) origin in mm
    isocentre,  # (x, y, z) rotation center in mm
    degrees_list,  # list or tensor of angles in degrees
    axis="z",  # "x", "y", or "z"
    bg_value=-1024.0,
):
    grid = make_grid(
        tensor_img, spacing, origin, isocentre, degrees_list, axis, bg_value
    )
    rotated_tensor = apply_grid(tensor_img, grid, bg_value)

    return rotated_tensor.squeeze(1)


def rotate_pt_z(pt, isocentre, angle):
    """pt, isocentre: LPS from the JSON
    angle: degree to rotate
    returns: the LPS for the rotated pt
    """
    pt = torch.tensor(pt)
    isocentre = torch.tensor(isocentre)
    angle = torch.tensor(angle)

    rad = torch.deg2rad(angle)
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)

    x, y, z = pt - isocentre

    R = torch.tensor([[cos_a, -sin_a, 0, 0], [sin_a, cos_a, 0, 0], [0, 0, 1, 0]])

    return torch.matmul(R, torch.tensor([x, y, z, 1])) + isocentre

