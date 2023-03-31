from torch import nn
from torchvision.models import resnet18
import torch
from einops import rearrange


class WpodNet(nn.Module):

    def __init__(self):
        super(WpodNet, self).__init__()
        resnet = resnet18(True)
        backbone = list(resnet.children())
        self.backbone = nn.Sequential(
            nn.BatchNorm2d(num_features = 3),
            *backbone[:3],
            *backbone[4:8],
        )
        self.detection = nn.Conv2d(in_channels=512, out_channels=8,
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        features = self.backbone(x)
        out = self.detection(features)
        out = rearrange(out, 'n c h w -> n h w c')
        return out


if __name__ == '__main__':
    m = WpodNet()
    x = torch.randn(32, 3, 720, 720)
    print(m)
    print(m(x).shape)
    #torch.randn(32, 3, 720, 720) -> torch.Size([32, 45, 45, 8])