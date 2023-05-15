import torch

import envs.base_env as base_env

class ReturnTracker():
    def __init__(self, device):
        num_envs = 1
        self._device = device
        self._episodes = 0
        self._mean_return = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_ep_len = torch.zeros([1], device=device, dtype=torch.float32)

        self._return_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._ep_len_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        self._eps_per_env_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        return

    def get_mean_return(self):
        return self._mean_return

    def get_mean_ep_len(self):
        return self._mean_ep_len

    def get_episodes(self):
        return self._episodes

    def get_eps_per_env(self):
        return self._eps_per_env_buf

    def reset(self):
        self._episodes = 0
        self._eps_per_env_buf[:] = 0

        self._mean_return[:] = 0.0
        self._mean_ep_len[:] = 0.0

        self._return_buf[:] = 0.0
        self._ep_len_buf[:] = 0
        return

    def update(self, reward, done):
        assert(reward.shape == self._return_buf.shape)
        assert(done.shape == self._return_buf.shape)

        self._return_buf += reward
        self._ep_len_buf += 1

        reset_mask = done != base_env.DoneFlags.NULL.value
        reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
        num_resets = len(reset_ids)

        if (num_resets > 0):
            new_mean_return = torch.mean(self._return_buf[reset_ids])
            new_mean_ep_len = torch.mean(self._ep_len_buf[reset_ids].type(torch.float))

            new_count = self._episodes + num_resets
            w_new = float(num_resets) / new_count
            w_old = float(self._episodes) / new_count

            self._mean_return = w_new * new_mean_return + w_old * self._mean_return
            self._mean_ep_len = w_new * new_mean_ep_len + w_old * self._mean_ep_len
            self._episodes += num_resets

            self._return_buf[reset_ids] = 0.0
            self._ep_len_buf[reset_ids] = 0
            self._eps_per_env_buf[reset_ids] += 1

        return