import enum
import numpy as np
import torch

class DistributionCategoricalBuilder(torch.nn.Module):
    def __init__(self, in_size, out_size, init_output_scale=0.01):
        super().__init__()
        self._build_params(in_size, out_size, init_output_scale)
        return

    def _build_params(self, in_size, out_size, init_output_scale):
        self._logit_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._logit_net.weight, -init_output_scale, init_output_scale)
        return

    def forward(self, input):
        logits = self._logit_net(input)
        dist = DistributionCategorical(logits=logits)
        return dist


class DistributionCategorical(torch.distributions.Categorical):
    def __init__(self, logits):
        logits = logits.unsqueeze(-2)
        super().__init__(logits=logits)
        return
    
    @property
    def mode(self):
        arg_max = torch.argmax(self.logits, dim=-1)
        return arg_max

    def sample(self):
        x = super().sample()
        return x

    def log_prob(self, x):
        logp = super().log_prob(x)
        logp = logp.squeeze(-1)
        return logp
    
    def entropy(self):
        ent = super().entropy()
        ent = ent.squeeze(-1)
        return ent

    def param_reg(self):
        # only regularize mean, covariance is regularized via entropy reg
        reg = torch.sum(torch.square(self.logits), dim=-1)
        reg = reg.squeeze(-1)
        return reg
