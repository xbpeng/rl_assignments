import abc
import enum
import gym
import numpy as np
import os
import time
import torch

import envs.base_env as base_env
import learning.experience_buffer as experience_buffer
import learning.normalizer as normalizer
import learning.return_tracker as return_tracker
import util.tb_logger as tb_logger

import learning.distribution_gaussian_diag as distribution_gaussian_diag

class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class BaseAgent(torch.nn.Module):
    NAME = "base"

    def __init__(self, config, env, device):
        super().__init__()

        self._env = env
        self._device = device
        self._iter = 0
        self._sample_count = 0
        self._load_params(config)

        self._build_normalizers()
        self._build_model(config)
        self._build_optimizer(config)

        self._build_exp_buffer(config)
        self._build_return_tracker()
        
        self._mode = AgentMode.TRAIN
        self._curr_obs = None
        self._curr_info = None

        self.to(self._device)

        return

    def train_model(self, max_samples, out_model_file, int_output_dir, log_file):
        start_time = time.time()

        self._curr_obs, self._curr_info = self._env.reset()
        self._logger = self._build_logger(log_file)
        self._init_train()

        test_info = None
        while self._sample_count < max_samples:
            train_info = self._train_iter()
            
            output_iter = (self._iter % self._iters_per_output == 0)
            if (output_iter):
                test_info = self.test_model(self._test_episodes)
            
            self._sample_count = self._update_sample_count()
            self._log_train_info(train_info, test_info, start_time) 
            self._logger.print_log()

            if (output_iter):
                self._logger.write_log()
                self._output_train_model(self._iter, out_model_file, int_output_dir)

                self._train_return_tracker.reset()
                self._curr_obs, self._curr_info = self._env.reset()
            
            self._iter += 1

        return

    def test_model(self, num_episodes):
        self.eval()
        self.set_mode(AgentMode.TEST)

        self._curr_obs, self._curr_info = self._env.reset()
        test_info = self._rollout_test(num_episodes)

        return test_info
    
    def get_action_size(self):
        a_space = self._env.get_action_space()
        if (isinstance(a_space, gym.spaces.Box)):
            a_size = np.prod(a_space.shape)
        elif (isinstance(a_space, gym.spaces.Discrete)):
            a_size = 1
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return a_size
    
    def set_mode(self, mode):
        self._mode = mode
        if (self._mode == AgentMode.TRAIN):
            self._env.set_mode(base_env.EnvMode.TRAIN)
        elif (self._mode == AgentMode.TEST):
            self._env.set_mode(base_env.EnvMode.TEST)
        else:
            assert(False), "Unsupported agent mode: {}".format(mode)
        return

    def save(self, out_file):
        state_dict = self.state_dict()
        torch.save(state_dict, out_file)
        return

    def load(self, in_file):
        state_dict = torch.load(in_file, map_location=self._device)
        self.load_state_dict(state_dict)
        return

    def _load_params(self, config):
        self._discount = config["discount"]
        self._steps_per_iter = config["steps_per_iter"]
        self._iters_per_output = config["iters_per_output"]
        self._test_episodes = config["test_episodes"]
        self._normalizer_samples = config.get("normalizer_samples", np.inf)
        return

    @abc.abstractmethod
    def _build_model(self, config):
        return

    def _build_normalizers(self):
        obs_shape = self._env.compute_obs_shape()
        self._obs_norm = normalizer.Normalizer(obs_shape, device=self._device)
        self._a_norm = self._build_action_normalizer()
        return
    
    def _build_action_normalizer(self):
        a_space = self._env.get_action_space()

        if (isinstance(a_space, gym.spaces.Box)):
            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device)
            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, dtype=a_mean.dtype)
        elif (isinstance(a_space, gym.spaces.Discrete)):
            a_mean = torch.tensor([0], device=self._device, dtype=torch.long)
            a_std = torch.tensor([1], device=self._device, dtype=torch.long)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
                                                 init_std=a_std, eps=0, dtype=a_mean.dtype)
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return a_norm

    def _build_optimizer(self, config):
        lr = float(config["learning_rate"])
        params = list(self.parameters())
        params = [p for p in params if p.requires_grad]
        self._optimizer = torch.optim.SGD(params, lr, momentum=0.9)
        return
    
    def _build_exp_buffer(self, config):
        buffer_length = self._get_exp_buffer_length()
        self._exp_buffer = experience_buffer.ExperienceBuffer(buffer_length=buffer_length, device=self._device)

        obs_shape = self._env.compute_obs_shape()
        obs_buffer = torch.zeros([buffer_length] + obs_shape, device=self._device, dtype=torch.float)
        self._exp_buffer.add_buffer("obs", obs_buffer)

        next_obs_buffer = torch.zeros_like(obs_buffer)
        self._exp_buffer.add_buffer("next_obs", next_obs_buffer)

        action_size = self.get_action_size()
        action_dtype = self._get_action_dtype()
        action_buffer = torch.zeros([buffer_length, action_size], device=self._device, dtype=action_dtype)
        self._exp_buffer.add_buffer("action", action_buffer)
        
        reward_buffer = torch.zeros([buffer_length], device=self._device, dtype=torch.float)
        self._exp_buffer.add_buffer("reward", reward_buffer)
        
        done_buffer = torch.zeros([buffer_length], device=self._device, dtype=torch.int)
        self._exp_buffer.add_buffer("done", done_buffer)

        return

    def _build_return_tracker(self):
        self._train_return_tracker = return_tracker.ReturnTracker(self._device)
        return

    @abc.abstractmethod
    def _get_exp_buffer_length(self):
        return 0
    
    def _get_action_dtype(self):
        dtype = None
        a_space = self._env.get_action_space()
        if (isinstance(a_space, gym.spaces.Box)):
            dtype = torch.float32
        elif (isinstance(a_space, gym.spaces.Discrete)):
            dtype = torch.long
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)
        return dtype

    def _build_logger(self, log_file):
        log = tb_logger.TBLogger()
        log.set_step_key("Samples")
        log.configure_output_file(log_file)
        return log

    def _update_sample_count(self):
        return self._exp_buffer.get_total_samples()

    def _init_train(self):
        self._iter = 0
        self._sample_count = 0
        self._exp_buffer.clear()
        self._train_return_tracker.reset()
        return

    def _train_iter(self):
        self._init_iter()
        
        self.eval()
        self.set_mode(AgentMode.TRAIN)
        self._rollout_train(self._steps_per_iter)
        
        data_info = self._build_train_data()
        train_info = self._update_model()
        
        if (self._need_normalizer_update()):
            self._update_normalizers()
        
        info = {**train_info, **data_info}
        
        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["episodes"] = self._train_return_tracker.get_episodes()
        
        return info

    def _init_iter(self):
        return

    def _rollout_train(self, num_steps):
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(r, done)
            self._record_data_post_step(next_obs, r, done, next_info)
            
            self._curr_obs = next_obs

            if done != base_env.DoneFlags.NULL.value:  
                self._curr_obs, self._curr_info = self._env.reset()
            self._exp_buffer.inc()

        return
    
    def _rollout_test(self, num_episodes):
        sum_rets = 0.0
        sum_ep_lens = 0
        
        self._curr_obs, self._curr_info = self._env.reset()

        for e in range(num_episodes):
            curr_ret = 0.0
            curr_ep_len = 0
            while True:
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)
                self._curr_obs, r, done, self._curr_info = self._step_env(action)

                curr_ret += r.item()
                curr_ep_len += 1

                if done != base_env.DoneFlags.NULL.value:
                    sum_rets += curr_ret
                    sum_ep_lens += curr_ep_len
                    self._curr_obs, self._curr_info = self._env.reset()
                    break

        
        mean_return = sum_rets / max(1, num_episodes)
        mean_ep_len = sum_ep_lens / max(1, num_episodes)

        test_info = {
            "mean_return": mean_return,
            "mean_ep_len": mean_ep_len,
            "episodes": num_episodes
        }
        return test_info

    @abc.abstractmethod
    def _decide_action(self, obs, info):
        a = None
        a_info = dict()
        return a, a_info

    def _step_env(self, action):
        obs, r, done, info = self._env.step(action)
        return obs, r, done, info

    def _record_data_pre_step(self, obs, info, action, action_info):
        self._exp_buffer.record("obs", obs)
        self._exp_buffer.record("action", action)

        self._obs_norm.record(obs.unsqueeze(0))
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        self._exp_buffer.record("next_obs", next_obs)
        self._exp_buffer.record("reward", r)
        self._exp_buffer.record("done", done)
        return

    def _reset_done_envs(self, done):
        done_indices = (done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False)
        done_indices = torch.flatten(done_indices)
        obs, info = self._env.reset(done_indices)
        return obs, info

    def _need_normalizer_update(self):
        return self._sample_count < self._normalizer_samples

    def _update_normalizers(self):
        self._obs_norm.update()
        return

    def _build_train_data(self):
        return dict()

    @abc.abstractmethod
    def _update_model(self):
        return

    def _compute_succ_val(self):
        r_succ = self._env.get_reward_succ()
        val_succ = r_succ / (1.0 - self._discount)
        return val_succ
    
    def _compute_fail_val(self):
        r_fail = self._env.get_reward_fail()
        val_fail = r_fail / (1.0 - self._discount)
        return val_fail

    def _compute_val_bound(self):
        r_min, r_max = self._env.get_reward_bounds()
        val_min = r_min / (1.0 - self._discount)
        val_max = r_max / (1.0 - self._discount)
        return val_min, val_max

    def _log_train_info(self, train_info, test_info, start_time):
        wall_time = (time.time() - start_time) / (60 * 60) # store time in hours
        self._logger.log("Iteration", self._iter, collection="1_Info")
        self._logger.log("Wall_Time", wall_time, collection="1_Info")
        self._logger.log("Samples", self._sample_count, collection="1_Info")
        
        test_return = test_info["mean_return"]
        test_ep_len = test_info["mean_ep_len"]
        test_eps = test_info["episodes"]
        self._logger.log("Test_Return", test_return, collection="0_Main")
        self._logger.log("Test_Episode_Length", test_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Test_Episodes", test_eps, collection="1_Info", quiet=True)

        train_return = train_info.pop("mean_return")
        train_ep_len = train_info.pop("mean_ep_len")
        train_eps = train_info.pop("episodes")
        self._logger.log("Train_Return", train_return, collection="0_Main")
        self._logger.log("Train_Episode_Length", train_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Train_Episodes", train_eps, collection="1_Info", quiet=True)

        for k, v in train_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v)
            
        return
    
    def _compute_action_bound_loss(self, norm_a_dist):
        loss = None
        action_space = self._env.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            a_low = action_space.low
            a_high = action_space.high
            valid_bounds = np.all(np.isfinite(a_low)) and np.all(np.isfinite(a_high))

            if (valid_bounds):
                assert(isinstance(norm_a_dist, distribution_gaussian_diag.DistributionGaussianDiag))
                # assume that actions have been normalized between [-1, 1]
                bound_min = -1
                bound_max = 1
                violation_min = torch.clamp_max(norm_a_dist.mode - bound_min, 0.0)
                violation_max = torch.clamp_min(norm_a_dist.mode - bound_max, 0)
                violation = torch.sum(torch.square(violation_min), dim=-1) \
                            + torch.sum(torch.square(violation_max), dim=-1)
                loss = violation

        return loss

    def _output_train_model(self, iter, out_model_file, int_output_dir):
        self.save(out_model_file)

        if (int_output_dir != ""):
            int_model_file = os.path.join(int_output_dir, "model_{:010d}.pt".format(iter))
            self.save(int_model_file)
        return
