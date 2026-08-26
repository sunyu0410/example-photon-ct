import torch

from tqdm.auto import tqdm
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from rotate import rotate_image_new
from geometry import get_source_location_mm, MLC
from bev_torch import (
    make_grid_5d,
    get_bev_torch,
    mm2idx,
    cal_scales,
    draw_iso_mlc,
)


def get_bbox(arr, margin=5):
    """Get the bounding box of an np mask
    results in np array as (zmin, ymin, xmin), (zmax, ymax, xmax)
    """

    shape = np.array(arr.shape)
    idx = np.array(np.where(arr > 0))

    # Safety check
    min_idx = np.clip(idx.min(1) - margin, 0, shape)
    max_idx = np.clip(idx.max(1) + margin, 0, shape)

    return np.stack([min_idx, max_idx])


# Index(['gantry_angle', 'mlc_left_int_mm', 'mlc_right_int_mm', 'cp_uuid',
#        'output_info.minimum_cutoff', 'output_info.output_file_idx',
#        'output_info.idx_in_output', 'image_file_idx', 'anatomical_region',
#        'beams.SAD', 'beams.iso_center', 'beams.num_mlc_leaf_pairs'],
#       dtype='str')


class InferenceBeamData(Dataset):
    def __init__(self, group):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.group = group
        self.len = group.shape[0]

        # Parse the group df
        self.input_idx = group["image_file_idx"].tolist()[0]
        self.isocentre = group["beams.iso_center"].tolist()[0]
        self.sad = group["beams.SAD"].tolist()[0]

        self.angles = torch.tensor(group["gantry_angle"].to_numpy())
        self.mlc_l = torch.tensor(np.stack(group["mlc_left_int_mm"].values))
        self.mlc_r = torch.tensor(np.stack(group["mlc_right_int_mm"].values))
        self.cutoff = torch.tensor(np.stack(group["output_info.minimum_cutoff"].values))

        
        self.img_sitk = self.load_input_by_index(self.input_idx) # loads the first image
        
        self.img = torch.tensor(sitk.GetArrayFromImage(self.img_sitk)).to(torch.float16)

        self.rot_input_tensor = (
            torch.randn((30,) + self.img_sitk.GetSize()[::-1])
            .to(torch.float32)
            .to(self.device)
        )

        self.img_rot = torch.zeros((self.len,) + self.img.shape)
        self.rotate_img()
        self.bev = self.cal_bev()

    def load_input_by_index(self, idx):
        location = INPUT_PATH / f"images/{INPUT_DIR_BASE}-{idx + 1}"
        filepath = location.glob("*.mha")[0]
        return sitk.ReadImage(filepath)

    def rotate_img(self):

        for i in tqdm(
            range(0, self.len // 30 + bool(self.len % 30)), desc="Rotating img"
        ):
            # Get the length of batch, it can be less than 30
            angles = self.angles[i * 30 : (i + 1) * 30]
            n = len(angles)

            # Rotate the img
            self.rot_input_tensor[:n] = self.img.expand(n, -1, -1, -1)
            self.img_rot[30 * i : (i + 1) * 30] = rotate_image_new(
                self.rot_input_tensor[:n],  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                self.img_sitk.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                self.img_sitk.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                self.isocentre,  # Can now be a torch.Tensor or list/tuple
                -angles,  # Can now be a torch.Tensor of angles on GPU
                axis="z",
                bg_value=-1024,
            ).cpu()

    def cal_bev(self, n_batch=30):

        def _cal_scales(self):
            src_mm = get_source_location_mm(self.isocentre, 0, self.sad)
            z_scales = torch.tensor(cal_scales(self.img_sitk, self.isocentre, src_mm))[
                ..., None, None
            ]
            return z_scales

        def _cal_mlc_2d(self):
            # Get the 2d bev at isocentre for each control point
            bev_iso_list = []
            for i in range(self.len):
                mlc = MLC.get_mlc_segs_mm(self.mlc_l[i], self.mlc_r[i], self.isocentre)
                bev_iso = draw_iso_mlc(self.img_sitk, mlc)  # (nx, nz) -> (246, 249)
                bev_iso_list.append(bev_iso)
            bev_iso_list = np.stack(bev_iso_list, axis=0)
            mlc_masks_2d = torch.tensor(bev_iso_list).to(torch.float32)
            return mlc_masks_2d

        # Cal z_scales
        z_scales = _cal_scales(self)
        mlc_masks_2d = _cal_mlc_2d(self)

        isocentre_idx = mm2idx(self.img_sitk, [self.isocentre])[0]

        # Set the batch number (# of images to process) and get the grid
        # The grid can be reused for each beam
        grid_5d = make_grid_5d(n_batch, isocentre_idx, z_scales, self.img_sitk).to(
            self.device
        )

        print("Grid created")

        # Get the beam path by batch
        bevs = []

        for i in tqdm(
            range(0, self.len // 30 + bool(self.len % 30)),
            desc="Calulating BEV beam path",
        ):
            batch = mlc_masks_2d[i * 30 : (i + 1) * 30]
            n = len(batch)
            bevs.append(get_bev_torch(batch, grid_5d[:n]))
        bevs = torch.cat(bevs, dim=0)

        return bevs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.img_rot[idx]
        bev = self.bev[idx]

        mask = img > -1024
        bev = bev * mask
        bounds = get_bbox(bev.numpy(), margin=3)
        loc = tuple([slice(*i) for i in zip(*bounds.tolist())])  # torch requires tuple

        return (
            img[loc], bev[loc], mask[loc], loc,
        )

    def inference(self, model):
        cp_preds = []
        for img, bev, mask, loc in tqdm(d):
            with torch.no_grad():
                box_pred = model(img, bev, mask).cpu()
                box_pred[bev==0] = 0 
                cp_pred = torch.zeros_like(d.img)
                cp_pred[loc] = box_pred
                cp_preds.append(cp_pred)
        self.preds = torch.stack(cp_preds, dim=0)

        self.preds_back = torch.zeros_like(self.preds)

        # Rotate it to the gantry angle
        for i in tqdm(
            range(0, self.len // 30 + bool(self.len % 30)), desc="Rotating preds"
        ):
            # Get the length of batch, it can be less than 30
            angles = self.angles[i * 30 : (i + 1) * 30]
            n = len(angles)

            # Rotate the img
            self.rot_input_tensor[:n] = self.preds[:n]
            self.preds_back[30 * i : (i + 1) * 30] = rotate_image_new(
                self.rot_input_tensor[:n],  # Shape: (D, H, W) or (1, 1, D, H, W) on GPU
                self.img_sitk.GetSpacing(),  # Can now be a torch.Tensor or list/tuple
                self.img_sitk.GetOrigin(),  # Can now be a torch.Tensor or list/tuple
                self.isocentre,  # Can now be a torch.Tensor or list/tuple
                angles,  # Can now be a torch.Tensor of angles on GPU
                axis="z",
                bg_value=0,
            ).cpu()

        return self.preds_back

if __name__ == "__main__":
    import pandas as pd
    import json

    from models import BeamNet
    model = BeamNet()
    model.load_state_dict(torch.load('model_weights.pth'))

    metadata = json.load(open("../metadata/stacked-photon-beam-level-metadata.txt"))
    df = pd.json_normalize(
        metadata, 
        record_path=['beams', 'control_points'], 
        meta=[
            'image_file_idx', 
            'anatomical_region', 
            ['beams', 'SAD'], 
            ['beams', 'iso_center'], 
            ['beams', 'num_mlc_leaf_pairs'],
        ]
    )
    groups = df.groupby('output_info.output_file_idx')

    
    for i in range(groups.ngroups):
        print(f'Processing Group {i}')
        d = InferenceBeamData(groups.get_group(i))
        out = d.inference(model)

        # Scale it back 1e-5
        out = out * 1e-5

        # Save to .mha e.g. (x, x, x, 40)
        preds_sitk = []
        for t in out.unbind(dim=0):
            pred_sitk = sitk.GetImageFromArray(t.float().numpy())
            pred_sitk.CopyInformation(d.img_sitk)
            preds_sitk.append(pred_sitk)
        stacked = sitk.JoinSeries(preds_sitk)

        sitk.WriteImage(stacked, output_dir / "output.mha", useCompression=False)

        break

    
