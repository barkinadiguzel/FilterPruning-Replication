import torch
import torch.nn as nn

def count_conv_flops(conv, input_shape):
    Cin, Hin, Win = input_shape
    Cout = conv.out_channels
    K_h, K_w = conv.kernel_size
    stride_h, stride_w = conv.stride
    pad_h, pad_w = conv.padding

    Hout = (Hin + 2 * pad_h - K_h) // stride_h + 1
    Wout = (Win + 2 * pad_w - K_w) // stride_w + 1

    flops = 2 * Cout * Cin * K_h * K_w * Hout * Wout
    return flops, (Cout, Hout, Wout)


def count_model_flops(model, input_size):
    total_flops = 0
    current_shape = input_size

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            flops, current_shape = count_conv_flops(layer, current_shape)
            total_flops += flops

    return total_flops
