import torch

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util

class ExpertModel(base_model.BaseModel):
    def __init__(self, config, env):
        super().__init__(config, env)

        self._build_nets(config, env)
        return

    def eval_actor(self, obs):
        h = self._actor_layers(obs)
        a_dist = self._action_dist(h)
        return a_dist
    
    def eval_critic(self, obs):
        h = self._critic_layers(obs)
        val = self._critic_out(h)
        return val
    
    def _build_nets(self, config, env):
        self._build_actor(config, env)
        self._build_critic(config, env)
        return

    def _build_actor(self, config, env):
        net_name = config["actor_net"]
        input_dict = self._build_actor_input_dict(env)
        self._actor_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                activation=self._activation)
        
        self._action_dist = self._build_action_distribution(config, env, self._actor_layers)
        return
    
    def _build_critic(self, config, env):
        net_name = config["critic_net"]
        input_dict = self._build_critic_input_dict(env)
        self._critic_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                 activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)
        return

    def _build_actor_input_dict(self, env):
        obs_space = env.get_obs_space()
        input_dict = {"obs": obs_space}
        return input_dict
    
    def _build_critic_input_dict(self, env):
        obs_space = env.get_obs_space()
        input_dict = {"obs": obs_space}
        return input_dict
