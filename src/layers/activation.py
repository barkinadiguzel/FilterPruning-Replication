import torch.nn as nn

def get_activation(name="relu"):
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation {name}")
