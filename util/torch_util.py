import numpy as np
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


def torch_dtype_to_numpy(torch_dtype):
    if (torch_dtype == torch.float32):
        numpy_dtype = np.float32
    elif (torch_dtype == torch.float64):
        numpy_dtype = np.float64
    elif (torch_dtype == torch.uint8):
        numpy_dtype = np.uint8
    elif (torch_dtype == torch.int64):
        numpy_dtype = np.int64
    else:
        assert(False), "Unsupported type {}".format(torch_dtype)
    return numpy_dtype

def numpy_dtype_to_torch(numpy_dtype):
    if (numpy_dtype == np.float32):
        torch_dtype = torch.float32
    elif (numpy_dtype == np.float64):
        torch_dtype = torch.float64
    elif (numpy_dtype == np.uint8):
        torch_dtype = torch.uint8
    elif (numpy_dtype == np.int64):
        torch_dtype = torch.int64
    else:
        assert(False), "Unsupported type {}".format(numpy_dtype)
    return torch_dtype

class UInt8ToFloat(torch.nn.Module):
    def forward(self, x):
        float_x = x.type(torch.float32) / 255.0
        return float_x