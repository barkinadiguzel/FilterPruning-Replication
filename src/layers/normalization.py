import torch.nn as nn

def get_norm(num_channels, use_bn=True):
    if use_bn:
        return nn.BatchNorm2d(num_channels)
    return nn.Identity()
