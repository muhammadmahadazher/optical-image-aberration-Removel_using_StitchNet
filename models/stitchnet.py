import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class StitchNet(nn.Module):
    def __init__(self):
        super().__init__()
        enc = torchvision.models.resnet18(weights=None)
        self.enc = nn.Sequential(*list(enc.children())[:-2])     # 512-ch maps
        # 1024 => 128 => 64 => 2 (dx,dy)
        self.flow = nn.Sequential(
            nn.Conv2d(1024,128,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  2,3,padding=1))

    def forward(self, ref, nbr):
        f1, f2 = self.enc(ref), self.enc(nbr)        # N×512×h×w each
        corr    = torch.cat([f1, f2], dim=1)         # N×1024×h×w
        flow_lo = self.flow(corr)                    # N×2×h×w
        flow    = F.interpolate(flow_lo, size=ref.shape[-2:], mode='bilinear', align_corners=False)

        # build sampling grid
        B, _, H, W = ref.shape
        y, x = torch.linspace(-1,1,H,device=ref.device), torch.linspace(-1,1,W,device=ref.device)
        grid = torch.stack(torch.meshgrid(y,x,indexing='ij'),-1).unsqueeze(0).repeat(B,1,1,1)
        warped = F.grid_sample(nbr, grid + flow.permute(0,2,3,1), mode='bilinear', align_corners=False)
        return warped, flow

