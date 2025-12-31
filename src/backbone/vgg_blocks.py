import torch.nn as nn
from src.layers.conv_layer import PrunableConv2d
from src.layers.activation import get_activation

def vgg_block(in_ch, out_ch, n_convs=2):
    layers = []
    for i in range(n_convs):
        layers.append(
            PrunableConv2d(
                in_ch if i == 0 else out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            )
        )
        layers.append(get_activation())
    return nn.Sequential(*layers)
