import torch.nn as nn
from src.backbone.vgg_blocks import vgg_block

class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([
            vgg_block(3, 64),
            vgg_block(64, 128),
            vgg_block(128, 256)
        ])

    def forward(self, x):
        for block in self.features:
            x = block(x)
        return x
