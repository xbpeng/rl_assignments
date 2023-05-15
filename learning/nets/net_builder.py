import torch

from learning.nets import *

def build_net(net_name, input_dict, activation=torch.nn.ReLU):
    if (net_name in globals()):
        net_func = globals()[net_name]
        net, info = net_func.build_net(input_dict, activation)
    else:
        assert(False), "Unsupported net: {}".format(net_name)
    return net, info
