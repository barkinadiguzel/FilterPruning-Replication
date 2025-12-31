import torch

def l1_filter_score(conv_layer):
    weight = conv_layer.weight.data
    scores = torch.sum(torch.abs(weight), dim=(1, 2, 3))
    return scores
