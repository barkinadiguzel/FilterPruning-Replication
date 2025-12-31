import torch
import torch.nn as nn

def prune_conv_layer(conv, prune_idx):
    keep_idx = [i for i in range(conv.out_channels) if i not in prune_idx]

    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep_idx),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None)
    )

    new_conv.weight.data = conv.weight.data[keep_idx].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_idx].clone()

    return new_conv
