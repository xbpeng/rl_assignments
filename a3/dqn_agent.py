import gym
import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.dqn_model as dqn_model
import util.torch_util as torch_util

class DQNAgent(base_agent.BaseAgent):
    NAME = "DQN"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)

        buffer_size = config["exp_buffer_size"]
        self._exp_buffer_length = int(buffer_size)
        self._exp_buffer_length = max(self._exp_buffer_length, self._steps_per_iter)

        self._updates_per_iter = config["updates_per_iter"]
        self._batch_size = config["batch_size"]
        self._init_samples = config["init_samples"]
        self._tar_net_update_iters = config["tar_net_update_iters"]
        
        self._exp_anneal_samples = config.get("exp_anneal_samples", np.inf)
        self._exp_prob_beg = config.get("exp_prob_beg", 1.0)
        self._exp_prob_end = config.get("exp_prob_end", 1.0)

        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = dqn_model.DQNModel(model_config, self._env)

        self._tar_model = dqn_model.DQNModel(model_config, self._env)
        for param in self._tar_model.parameters():
            param.requires_grad = False

        self._sync_tar_model()
        return
    
    def _get_exp_buffer_length(self):
        return self._exp_buffer_length

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        qs = self._model.eval_q(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            a = self._sample_action(qs)
        elif (self._mode == base_agent.AgentMode.TEST):
            a = torch.argmax(qs, dim=-1)
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)

        a_info = {}
        return a, a_info

    def _init_train(self):
        super()._init_train()
        self._collect_init_samples(self._init_samples)
        return

    def _collect_init_samples(self, samples):
        self.eval()
        self.set_mode(base_agent.AgentMode.TRAIN)
        self._rollout_train(samples)
        return

    def _update_model(self):
        self.train()
        
        train_info = dict()

        for i in range(self._updates_per_iter):
            batch = self._exp_buffer.sample(self._batch_size)
            loss_info = self._compute_loss(batch)
                
            self._optimizer.zero_grad()
            loss = loss_info["loss"]
            loss.backward()
            self._optimizer.step()
            
            torch_util.add_torch_dict(loss_info, train_info)
        
        torch_util.scale_torch_dict(1.0 / self._updates_per_iter, train_info)

        if (self._iter % self._tar_net_update_iters == 0):
            self._sync_tar_model()

        return train_info
    
    def _log_train_info(self, train_info, test_info, start_time):
        super()._log_train_info(train_info, test_info, start_time)
        self._logger.log("Exp_Prob", self._get_exp_prob())
        return

    def _compute_loss(self, batch):
        r = batch["reward"]
        done = batch["done"]
        a = batch["action"]
        norm_obs = self._obs_norm.normalize(batch["obs"])
        norm_next_obs = self._obs_norm.normalize(batch["next_obs"])
        
        tar_vals = self._compute_tar_vals(r, norm_next_obs, done)
        q_loss = self._compute_q_loss(norm_obs, a, tar_vals)
        
        info = {"loss": q_loss}
        return info
    

    
    def _get_exp_prob(self):
        '''
        TODO 1.1: Calculate the epsilon-greedy exploration probability given the current sample
        count. The exploration probability should start with a value of self._exp_prob_beg, and
        then linearly annealed to self._exp_prob_end over the course of self._exp_anneal_samples
        timesteps.
        '''

        # placeholder
        prob = 1.0

        return prob

    def _sample_action(self, qs):
        '''
        TODO 1.2: Sample actions according to the Q-values of each action (qs). Implement epsilon
        greedy exploration, where the probability of sampling a random action (epsilon) is specified
        by self._get_exp_prob(). With probability 1 - epsilon, greedily select the action with
        the highest Q-value. With probability epsilon, select a random action uniformly from the
        set of possible actions. The output (a) should be a tensor containing the index of the selected
        action.
        '''
        exp_prob = self._get_exp_prob()

        # placeholder
        a = torch.zeros(qs.shape[0], device=self._device, dtype=torch.int64)
        return a
    
    def _compute_tar_vals(self, r, norm_next_obs, done):
        '''
        TODO 1.3: Compute target values for updating the Q-function. The inputs consist of a tensor
        of rewards (r), normalized observations at the next timestep (norm_next_obs), and done flags
        (done) indicating if a timestep is the last timestep of an episode. The output (tar_vals)
        should be a tensor containing the target values for each sample. The target values should
        be calculated using the target model (self._tar_model), not the main model (self._model).
        The Q-function can be queried by using the method eval_q(norm_obs).
        '''
        
        # placeholder
        tar_vals = torch.zeros_like(r)

        return tar_vals

    def _compute_q_loss(self, norm_obs, a, tar_vals):
        '''
        TODO 1.4: Compute a loss for updating the Q-function. The inputs consist of a tensor of
        normalized observations (norm_obs), a tensor containing the indices of actions selected
        at each timestep (a), and target values for each timestep (tar_vals). The output (loss)
        should be a scalar tensor containing the loss for updating the Q-function.
        '''
        
        # placeholder
        loss = torch.zeros(1)
        
        return loss
    
    def _sync_tar_model(self):
        '''
        TODO 1.5: Update the target model by copying the parameters from the main model. The
        main model is given by self._model, and the target model is given by self._tar_model.
        HINT: self._model.parameters() can be used to retrieve a list of tensors containing
        the parameters of a model.
        '''
        
        return