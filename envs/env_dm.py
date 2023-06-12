import envs.base_env as base_env

from dm_control import suite
import dm_env
import gym
import multiprocessing
import numpy as np
import pygame
import torch

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 360

class DMEnv(base_env.BaseEnv):
    def __init__(self, config, device, visualize):
        super().__init__(visualize)
    
        self._device = device
        self._visualize = visualize
        self._time_limit = config['time_limit']
        self._mode = None
        
        self._env = self._build_dm_env(config)

        if (self._visualize):
            self._init_render()

        self.reset()

        return


    def reset(self):
        time_step = self._env.reset()
        obs = self._convert_obs(time_step.observation)
        return obs, {}


    def step(self, action):
        time_step = self._env.step(action.cpu().numpy())
    
        obs = self._convert_obs(time_step.observation)
        reward = 0 if time_step.reward == None else time_step.reward
        reward = torch.tensor([reward], device=self._device)
        done = self._compute_done(time_step.last()) 
        info = dict()
        
        if self._visualize:
            self._render()

        return obs, reward, done, info 

 
    def _build_dm_env(self, config):
        env_name = config['env_name']
        task_name = config['task']
        assert(env_name.startswith("dm_"))
        env_name = env_name[3:]
        dm_env = suite.load(env_name, task_name)
        return dm_env
    
    def _convert_obs(self, obs):
        obs_list = []
        for v in obs.values():
            if not isinstance(v, np.ndarray): 
                obs_list.append(np.array([v]))
            else:  
                obs_list.append(v)

        flat_obs = np.concatenate(obs_list, axis=-1)
        flat_obs = flat_obs.astype(np.float32)
        flat_obs = torch.from_numpy(flat_obs)
        flat_obs = flat_obs.to(device=self._device)

        return  flat_obs

    def _compute_done(self, last):
        if self._env._physics.time() >= self._time_limit: 
            done = torch.tensor([base_env.DoneFlags.TIME.value], device=self._device)
        elif last: 
            done = torch.tensor([base_env.DoneFlags.FAIL.value], device=self._device)
        else:
            done = torch.tensor([base_env.DoneFlags.NULL.value], device=self._device)
        return done

    def get_action_space(self):
        action_spec = self._env.action_spec()
        if (isinstance(action_spec, dm_env.specs.BoundedArray)):
            low = action_spec.minimum
            high = action_spec.maximum
            action_space = gym.spaces.Box(low=low, high=high, shape=low.shape)
        else:
            assert(False), "Unsupported DM action type"

        return action_space
  
    def get_reward_bounds(self):
        return (0.0, 1.0)

    def _init_render(self):
        self._clock = pygame.time.Clock()

        self._render_queue = multiprocessing.Queue()
        self._render_proc = multiprocessing.Process(target=render_worker, args=(SCREEN_WIDTH, SCREEN_HEIGHT, self._render_queue))
        self._render_proc.start()
        return

    def _render(self):
        im = self._env.physics.render(SCREEN_HEIGHT, SCREEN_WIDTH, camera_id=0)
        im = np.transpose(im, axes=[1, 0, 2])
        self._render_queue.put(im)

        fps = 1.0 / self._env.control_timestep()
        self._clock.tick(fps)
        return

   


def render_worker(screen_w, screen_h, render_queue):
    pygame.init()     
    window = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("DeepMind Control Suite")

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        im = render_queue.get()
        surface = pygame.pixelcopy.make_surface(im)
        window.blit(surface, (0, 0))
        pygame.display.update()

    pygame.quit()
    return
