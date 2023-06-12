import gym
import torch

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class DQNModel(base_model.BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._build_nets(config, env)
        return

    def eval_q(self, obs):
        h = self._q_layers(obs)
        q = self._q_out(h)
        return q
    
    def _build_nets(self, config, env):
        net_name = config["q_net"]
        q_init_output_scale = config["q_init_output_scale"]
        input_dict = self._build_q_input_dict(env)
        
        a_space = env.get_action_space()
        assert(isinstance(a_space, gym.spaces.Discrete))
        num_actions = a_space.n

        self._q_layers, _ = net_builder.build_net(net_name, input_dict,
                                                  activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._q_layers)
        
        self._q_out = torch.nn.Linear(layers_out_size, num_actions)
        torch.nn.init.uniform_(self._q_out.weight, -q_init_output_scale, q_init_output_scale)
        torch.nn.init.zeros_(self._q_out.bias)
        
        return
    
    def _build_q_input_dict(self, env):
        obs_space = env.get_obs_space()
        input_dict = {"obs": obs_space}
        return input_dict