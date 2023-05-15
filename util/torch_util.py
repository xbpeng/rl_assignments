import torch

def add_torch_dict(in_dict, out_dict):
    for k, v in in_dict.items():
        if (v.requires_grad):
            v = v.detach()

        if (k in out_dict):
            out_dict[k] += v
        else:
            out_dict[k] = v
    return
        
def scale_torch_dict(scale, out_dict):
    for k in out_dict.keys():
        out_dict[k] *= scale
    return

def calc_layers_out_size(layers):
    modules = list(layers.modules())
    for m in reversed(modules):
        if hasattr(m, "out_features"):
            out_size = m.out_features
            break
    return out_size