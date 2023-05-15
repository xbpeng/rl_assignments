import abc
import enum
import numpy as np

class EnvMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class DoneFlags(enum.Enum):
    NULL = 0
    FAIL = 1
    SUCC = 2
    TIME = 3

class BaseEnv(abc.ABC):
    NAME = "base"

    def __init__(self, visualize):
        self._mode = EnvMode.TRAIN
        self._visualize = visualize
        self._action_space = None
        return
    
    @abc.abstractmethod
    def reset(self, env_ids=None):
        return
    
    @abc.abstractmethod
    def step(self, action):
        return
    
    def compute_obs_shape(self):
        obs, obs_dict = self.reset()
        obs_shape = list(obs.shape)
        return obs_shape
    
    def get_action_space(self):
        return self._action_space

    def set_mode(self, mode):
        self._mode = mode
        return

    def get_num_envs(self):
        return int(1)

    def get_reward_bounds(self):
        return (-np.inf, np.inf)

    def get_reward_fail(self):
        return 0.0

    def get_reward_succ(self):
        return 0.0

    def get_visualize(self):
        return self._visualize
