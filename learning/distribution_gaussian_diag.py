import enum
import numpy as np
import torch

class StdType(enum.Enum):
    FIXED = 0
    CONSTANT = 1
    VARIABLE = 2

class DistributionGaussianDiagBuilder(torch.nn.Module):
    def __init__(self, in_size, out_size, std_type, init_std, init_output_scale=0.01):
        super().__init__()
        self._std_type = std_type

        self._build_params(in_size, out_size, init_std, init_output_scale)
        return

    def _build_params(self, in_size, out_size, init_std, init_output_scale):
        self._mean_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._mean_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._mean_net.bias)

        logstd = np.log(init_std)
        if (self._std_type == StdType.FIXED):
            self._logstd_net = torch.nn.Parameter(torch.zeros(out_size, requires_grad=False, dtype=torch.float32), requires_grad=False)
            torch.nn.init.constant_(self._logstd_net, logstd)
        elif (self._std_type == StdType.CONSTANT):
            self._logstd_net = torch.nn.Parameter(torch.zeros(out_size, requires_grad=True, dtype=torch.float32), requires_grad=True)
            torch.nn.init.constant_(self._logstd_net, logstd)
        elif (self._std_type == StdType.VARIABLE):
            self._logstd_net = torch.nn.Linear(in_size, out_size)
            torch.nn.init.uniform_(self._logstd_net.weight, -init_output_scale, init_output_scale) 
            torch.nn.init.constant_(self._logstd_net.bias, logstd)
        else:
            assert(False), "Unsupported StdType: {}".format(self._std_type)

        return

    def forward(self, input):
        mean = self._mean_net(input)

        if (self._std_type == StdType.FIXED or self._std_type == StdType.CONSTANT):
            logstd = torch.broadcast_to(self._logstd_net, mean.shape)
        elif (self._std_type == StdType.VARIABLE):
            logstd = self._logstd_net(input)
        else:
            assert(False), "Unsupported StdType: {}".format(self._std_type)

        dist = DistributionGaussianDiag(mean=mean, logstd=logstd)
        return dist

class DistributionGaussianDiag():
    def __init__(self, mean, logstd):
        self._mean = mean
        self._logstd = logstd
        self._std = torch.exp(self._logstd)
        self._dim = self._mean.shape[-1]
        return

    @property
    def stddev(self):
        return self._std
        
    @property
    def logstd(self):
        return self._logstd
        
    @property
    def mean(self):
        return self._mean
        
    @property
    def mode(self):
        return self._mean

    def sample(self):
        noise = torch.normal(torch.zeros_like(self._mean), torch.ones_like(self._std))
        x = self._mean + self._std * noise
        return x

    def log_prob(self, x):
        diff = x - self._mean
        logp = -0.5 * torch.sum(torch.square(diff / self._std), dim=-1)
        logp += -0.5 * self._dim * np.log(2.0 * np.pi) - torch.sum(self._logstd, dim=-1)
        return logp

    def entropy(self):
        ent = torch.sum(self._logstd, dim=-1)
        ent += 0.5 * self._dim * np.log(2.0 * np.pi * np.e)
        return ent
        
    def kl(self, other):
        assert(isinstance(other, GaussianDiagDist))
        other_var = torch.square(other.stddev)
        res = torch.sum(other.logstd - self._logstd + (torch.square(self._std) + torch.square(self._mean - other.mean)) / (2.0 * other_var), dim=-1)
        res += -0.5 * self._dim
        return res

    def param_reg(self):
        # only regularize mean, covariance is regularized via entropy reg
        reg = torch.sum(torch.square(self._mean), dim=-1)
        return reg