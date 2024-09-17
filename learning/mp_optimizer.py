import torch

import util.mp_util as mp_util

class MPOptimizer():
    CHECK_SYNC_STEPS = 1000

    def __init__(self, config, param_list):
        self._param_list = param_list
        self._grad_list = None
        self._optimizer = self._build_optimizer(config, param_list)
        self._steps = 0
        
        if (mp_util.enable_mp()):
            self._param_buffer = self._build_param_buffer()

        self.sync()
        return
    
    def step(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        
        if (mp_util.enable_mp()):
            self._aggregate_mp_grads()

        self._optimizer.step()
        
        if (mp_util.enable_mp() and (self.get_steps() % self.CHECK_SYNC_STEPS == 0)):
            assert(self._check_synced()), "Network parameters desynchronized"

        self._steps += 1
        return

    def get_steps(self):
        return self._steps

    def sync(self):
        with torch.no_grad():
            for param in self._param_list:
                global_param = mp_util.broadcast(param)
                param.copy_(global_param)
        return

    def _build_optimizer(self, config, param_list):
        lr = float(config["learning_rate"])
        optimizer_type = config["type"]
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(param_list, lr, momentum=0.9)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(param_list, lr)
        else:
            assert(False), "Unsupported optimizer type: " + optimizer_type
        return optimizer
    
    def _build_param_buffer(self):
        buffer = torch.nn.utils.parameters_to_vector(self._param_list).clone().detach()
        return buffer
    
    def _check_synced(self):
        synced = True
        for param in self._param_list:
            global_param = mp_util.broadcast(param)
            param_synced = torch.equal(param, global_param)
            if (not param_synced):
                synced = False
        
        device = self._param_list[0].device
        buffer = torch.tensor([synced], dtype=torch.int, device=device)
        mp_util.reduce_min(buffer)
        synced = buffer.item() != 0

        return synced

    def _aggregate_mp_grads(self):
        if (self._grad_list is None):
            self._grad_list = [p.grad for p in self._param_list]

        self._param_buffer[:] = torch.nn.utils.parameters_to_vector(self._grad_list)
        mp_util.reduce_inplace_mean(self._param_buffer)
        torch.nn.utils.vector_to_parameters(self._param_buffer, self._grad_list)
        return
