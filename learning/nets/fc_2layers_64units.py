import numpy as np
import torch

def build_net(input_dict, activation):
    layer_sizes = [64, 32]
    
    input_dim = np.sum([np.prod(curr_input.shape) for curr_input in input_dict.values()])
    
    in_size = input_dim
    layers = []
    for out_size in layer_sizes:
        curr_layer = torch.nn.Linear(in_size, out_size)
        torch.nn.init.zeros_(curr_layer.bias)

        layers.append(curr_layer)
        layers.append(activation())
        in_size = out_size

    net = torch.nn.Sequential(*layers)
    info = dict()

    return net, info
