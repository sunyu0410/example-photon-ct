import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class BeamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = smp.MAnet(
            encoder_name="resnet34",  
            encoder_weights=None,     
            in_channels=4,            
            classes=1,                      # Outputs 1 slice of continuous 3D Dose radiation map
            ).to(self.device)

    def forward(self, img, bev, mask):
        img = img.moveaxis(1,0).unsqueeze(1)
        bev = bev.moveaxis(1,0).unsqueeze(1)
        mask = mask.moveaxis(1,0).unsqueeze(1)

        img_norm = img/1000+1.024

        depth = torch.cumsum(mask, dim=0)
        wed = torch.cumsum(mask*img_norm, dim=0)

        x = torch.cat((img_norm, bev, depth, wed), dim=1).to(self.device)

        h, w = x.shape[2], x.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant')

        # Reverse the operations
        print(list(self.model.parameters())[0].device, x.device)
        out = self.model(x)[:,:,:h,:w]
        out = out.squeeze(1).moveaxis(0,1)
        
        return out


class LaplacianSmoothness2DLoss(nn.Module):
    def __init__(self, lambda_smooth=1e-4):
        """
        Parameters:
        -----------
        lambda_smooth : float
            The weight multiplier for the curvature penalty.
            Start small (e.g., 1e-5 or 1e-4) to balance it with your regression loss.
        """
        super().__init__()
        self.base_loss = nn.L1Loss() # Sharp absolute error baseline
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target):
        """
        pred, target shape: [Batch, Channels, Height, Width] (e.g., [B, 1, 250, 250])
        """
        # 1. Compute standard structural regression loss
        main_loss = self.base_loss(pred, target)

        # 2. Compute Second-Order Spatial Differences (Laplacian approximation)
        # Height (Y-axis) curvature: (y+2) - 2*(y+1) + (y)
        lap_y = torch.abs(pred[:, :, 2:, :] - 2 * pred[:, :, 1:-1, :] + pred[:, :, :-2, :]).mean()

        # Width (X-axis) curvature: (x+2) - 2*(x+1) + (x)
        lap_x = torch.abs(pred[:, :, :, 2:] - 2 * pred[:, :, :, 1:-1] + pred[:, :, :, :-2]).mean()

        # Combined 2D Laplacian curvature penalty
        total_lap = lap_y + lap_x

        # 3. Return balanced total loss
        return main_loss + (self.lambda_smooth * total_lap)