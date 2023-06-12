import envs.base_env as base_env
import envs.atari_wrappers as atari_wrappers

import gym
import numpy as np
import torch

class AtariEnv(base_env.BaseEnv):
    def __init__(self, config, device, visualize):
        super().__init__(visualize)
    
        self._device = device
        self._curr_obs = None
        self._curr_info = None
        
        self._env = self._build_atari_env(config, visualize)

        self.reset()

        return

    def reset(self):
        obs, info = self._env.reset()
        self._curr_obs = self._convert_obs(obs)
        self._curr_info = info
        return self._curr_obs, self._curr_info

    def step(self, action):
        a = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, info = self._env.step(a)
        self._curr_obs = self._convert_obs(obs)
        self._curr_info = info

        # train with clipped reward and then test with unclipped rewards
        if (self._mode is base_env.EnvMode.TRAIN):
            reward = reward[1] # clipped reward
        else:
            reward = reward[0] # unclipped reward

        reward = torch.tensor([reward], device=self._device)
        done = self._compute_done(terminated, truncated) 
        
        return self._curr_obs, reward, done, self._curr_info 

    def _build_atari_env(self, config, visualize):
        env_name = config["env_name"]
        max_timesteps = config.get("max_timesteps", None)
        assert(env_name.startswith("atari_"))

        env_name = env_name[6:]
        env_name = env_name + "NoFrameskip-v4"

        atari_env = atari_wrappers.make_atari(env_name, 
                                              max_episode_steps=max_timesteps,
                                              visualize=visualize)
        atari_env = atari_wrappers.wrap_deepmind(atari_env,
                                                 warp_frame=True,
                                                 frame_stack=True)
        
        return atari_env
 
    def _convert_obs(self, obs):
        num_frames = obs.count()
        frames = [np.expand_dims(obs.frame(i), axis=-3) for i in range(num_frames)]
        frames = np.concatenate(frames, axis=-3)
        
        obs = torch.tensor(frames, device=self._device)
        obs = obs.unsqueeze(0)

        return  obs

    def _compute_done(self, terminated, truncated):
        if (terminated): 
            done = torch.tensor([base_env.DoneFlags.FAIL.value], device=self._device)
        elif (truncated): 
            done = torch.tensor([base_env.DoneFlags.TIME.value], device=self._device)
        else:
            done = torch.tensor([base_env.DoneFlags.NULL.value], device=self._device)
        return done

    def get_obs_space(self):
        obs_space = self._env.observation_space
        obs_shape = list(obs_space.shape)
        obs_shape = obs_shape[-1:] + obs_shape[:-1]
        obs_space = gym.spaces.Box(
            low=obs_space.low.min(),
            high=obs_space.high.max(),
            shape=obs_shape,
            dtype=obs_space.dtype,
        )
        return obs_space

    def get_action_space(self):
        a_space = self._env.action_space
        a_space._shape = (1,)
        return a_space
  
    def get_reward_bounds(self):
        return (-1.0, 1.0)