import torch
from scipy import ndimage
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class MLPProcessor():
    def __init__(self, ct, bev, dose):
        self.ct = ct.clip(min=-1024)
        self.bev = bev
        self.dose = dose

        self.dim = 1
        self.ct_norm = self.ct/1000+1.024
        self.mask = self.ct > -1024

        self.depth = torch.cumsum(self.mask, dim=self.dim)
        self.wed = torch.cumsum(self.mask*self.ct_norm, dim=self.dim)
        self.red = self.cal_red()
        self.rings = self.cal_onion()
        self.keys = ['bev3', 'ring3', 'ring2', 'ring1']

        self.data = torch.stack([self.ct_norm, self.bev, self.depth, self.wed, self.red], dim=0).float()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def cal_red(self):
        red = torch.zeros_like(self.ct)
        red[self.ct<0] = 1 + 0.001*self.ct[self.ct<0]
        red[self.ct>=0] = 1 + 0.005*self.ct[self.ct>=0]
        red = torch.cumsum(self.mask*red, dim=self.dim)

        return red
    
    def cal_onion(self):
        bev3, bev2, bev1 = [torch.tensor(ndimage.binary_erosion(
            self.bev.numpy(), structure=None, iterations=i
        )).to(int) for i in (3,2,1)]

        ring1 = self.bev - bev1
        ring2 = bev1 - bev2
        ring3 = bev2 - bev3

        return dict(
            bev3 = bev3.bool(),
            ring3 = ring3.bool(),
            ring2 = ring2.bool(),
            ring1 = ring1.bool()
        )
    
    @staticmethod
    def sample_from_core(foreground_mask, to_sample):
        outside_mask = (1-foreground_mask).astype(np.uint8)
        distances, (nearest_z, nearest_y, nearest_x) = distance_transform_edt(
            outside_mask, return_indices=True
        )

        return to_sample[nearest_z, nearest_y, nearest_x], distances
    
    def prepare_ring_data(self, key, inner_dose=None, with_dose=False):

        ring_1d = self.rings[key].ravel()
        
        # Project the inner dose to the rest of volume
        if inner_dose is not None:
            border_dose, distances = self.sample_from_core((inner_dose!=0).numpy(), inner_dose)
            distances = torch.tensor(distances)
            inner_dose_data = torch.stack([border_dose, distances], dim=0).float()
            
            data = torch.cat([self.data, inner_dose_data], dim=0)
        else:
            data = self.data.clone()

        x = data.reshape((data.shape[0], -1)).moveaxis(1, 0)
        x = x[ring_1d!=0].to(self.device)

        if with_dose is True:
            y = self.dose.unsqueeze(0).reshape((1, -1)).moveaxis(1,0)
            y = y[ring_1d!=0].to(self.device)
            return x, y
        else:
            return x
        
    def get_xy(self):
        '''For TRAINING ONLY. The ring dose is from GT for training'''
        xs = []
        ys = []

        for key in self.keys:
            if key == 'bev3':
                x, y = self.prepare_ring_data(key, with_dose=True)
            else:
                # Create an inner dose using masked gt dose
                inner_dose = self.dose.clone()
                inner_dose[self.rings[key]==0] = 0
                x, y = self.prepare_ring_data(key, inner_dose=inner_dose, with_dose=True)

            xs.append(x)
            ys.append(y)
        return xs, ys
        

    def inference(self, model):
        '''The ring dose is from model prediction'''

        preds = {}
        inner_dose = torch.zeros_like(self.bev).float() # Accumalates the pred

        # bev3
        x = self.prepare_ring_data('bev3')
        preds['bev3'] = model.bev3(x)
        inner_dose[self.rings['bev3']!=0] = preds['bev3'].ravel().cpu()

        # ring3
        x = self.prepare_ring_data('ring3', inner_dose=inner_dose)
        preds['ring3'] = model.ring3(x)
        inner_dose[self.rings['ring3']!=0] = preds['ring3'].ravel().cpu()

        # ring2
        x = self.prepare_ring_data('ring2', inner_dose=inner_dose)
        preds['ring2'] = model.ring2(x)
        inner_dose[self.rings['ring2']!=0] = preds['ring2'].ravel().cpu()

        # ring1
        x = self.prepare_ring_data('ring1', inner_dose=inner_dose)
        preds['ring1'] = model.ring1(x)
        inner_dose[self.rings['ring1']!=0] = preds['ring1'].ravel().cpu()

        return inner_dose
        
    
    def get_full_map(self, preds):

        # Get the four pred maps
        pred_maps = [torch.zeros_like(self.dose).float() for i in range(4)]
        for pred_map, key, pred in zip(pred_maps, self.keys, preds):
            pred_map[self.rings[key]!=0] = pred.ravel()

        # Return the sum of them
        return torch.stack(pred_maps, dim=0).sum(0)


class SimpleMLP(nn.Module):
    def __init__(self, in_channel=5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.out(x)
        return out
    
class DoseModel():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bev3 = SimpleMLP(in_channel=5).to(self.device)
        self.ring3 = SimpleMLP(in_channel=7).to(self.device)
        self.ring2 = SimpleMLP(in_channel=7).to(self.device)
        self.ring1 = SimpleMLP(in_channel=7).to(self.device)

    def __call__(self, xs):
        models = [self.bev3, self.ring3, self.ring2, self.ring1]
        preds = [model(x) for model, x in zip(models, xs)]

        return preds

    def load_weight(self, weight_dir):
        weight_dir = Path(weight_dir)

        self.bev3.load_state_dict(torch.load(weight_dir / 'bev3.pth'))
        self.ring3.load_state_dict(torch.load(weight_dir / 'ring3.pth'))
        self.ring2.load_state_dict(torch.load(weight_dir / 'ring2.pth'))
        self.ring1.load_state_dict(torch.load(weight_dir / 'ring1.pth'))

        print('Weights loaded')

import numpy as np
from scipy.ndimage import gaussian_filter

def apply_3d_glow(volume, mask, sigma=2.0, glow_intensity=0.5, radius=None):
    """
    Applies a glowing effect around a 3D mask within a volume.

    Parameters:
    -----------
    volume : ndarray
        The original 3D image volume (normalized between 0.0 and 1.0).
    mask : ndarray (boolean or binary)
        The 3D binary mask where the signal is located.
    sigma : float or sequence of scalars
        Standard deviation for Gaussian kernel (controls glow width/radius).
    glow_intensity : float
        Multiplier for the glow brightness.
    """
    # Ensure inputs are floats for mathematical operations
    volume_float = volume.numpy().astype(np.float32)
    
    # Generate the 3D glow map by blurring the mask
    # This creates a smooth intensity falloff extending outside the mask boundaries
    glow_map = gaussian_filter(volume_float, sigma=sigma, radius=radius)

    # Suppress the glow *inside* the mask so it only appears around the edges
    # (Optional: remove this line if you want the mask interior to brighten up too)
    glow_map[mask > 0] = 0.0

    # Clip values to maintain valid intensity ranges (e.g., 0.0 to 1.0)
    return torch.tensor(volume_float + glow_map * glow_intensity)

def add_glow(pred, bev):
    '''pred, bev: in 3D'''
    pred[bev==0] = 0
    out = apply_3d_glow(pred, bev, sigma=3, glow_intensity=0.19)
    out = apply_3d_glow(out, bev, sigma=17, glow_intensity=0.10)

    return out

def infer_parallel(d, model, num_threads=8):
    n_cp = len(d)

    preds = torch.zeros((n_cp, *d.img.shape))

    def infer(idx):
        img, bev, mask, loc = d[idx]
        with torch.inference_mode():
            proc = MLPProcessor(img, bev, None)
            pred = proc.inference(model)
            
        preds[idx, *loc] = pred.detach().cpu()

        # Add the glow and mask outside the body
        mask = d.img_rot[idx] > -1024
        bev = d.bev[idx] * mask
        preds[idx] = add_glow(preds[idx], bev)
        preds[idx][mask==0] = 0


    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(infer, range(n_cp)),
                total=n_cp,
                desc="Inferencing",
            )
        )
        

    return preds

