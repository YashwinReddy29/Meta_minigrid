# envs.py
#
# Three env kinds:
#   - "empty"      -> MiniGrid-Empty-16x16-v0
#   - "walls"      -> MiniGrid-FourRooms-v0
#   - "obstacles"  -> MiniGrid-Dynamic-Obstacles-16x16-v0
#
# Observations: fully observable symbolic grid, flattened to 1D vector.
from gymnasium import RewardWrapper
from math import inf

class ShapedRewardWrapper(RewardWrapper):
    """
    Reward shaping for harder envs (walls, obstacles):

    - small negative step cost (-0.01),
    - bonus when agent gets closer to the goal,
    - extra bonus on successful termination.
    """

    def __init__(self, env, alpha: float = 0.05, step_penalty: float = 0.01, success_bonus: float = 0.5):
        super().__init__(env)
        self.alpha = alpha
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus
        self._last_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_dist = self._current_distance()
        return obs, info

    def _current_distance(self):
        # Manhattan distance between agent and goal if available
        env = self.env.unwrapped
        agent_pos = getattr(env, "agent_pos", None)
        goal_pos = getattr(env, "goal_pos", None)
        if agent_pos is None or goal_pos is None:
            return None
        return abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_dist = self._current_distance()
        shaped = 0.0

        if self._last_dist is not None and cur_dist is not None:
            # positive if we moved closer, negative if we moved away
            shaped = self.alpha * (self._last_dist - cur_dist)

        self._last_dist = cur_dist

        total = reward  # base env reward

        # time penalty every step
        total -= self.step_penalty

        # distance shaping
        total += shaped

        # extra bump on successful termination
        if terminated and reward > 0:
            total += self.success_bonus

        return obs, total, terminated, truncated, info

import gymnasium as gym
import numpy as np

from minigrid.wrappers import FullyObsWrapper

from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed


class FlatObsWrapper(ObservationWrapper):
    """
    Convert Minigrid's obs (Dict(image, mission) or just image Box)
    into a flat 1D float32 vector in [0, 1].
    """

    def __init__(self, env):
        super().__init__(env)
        orig_space = env.observation_space

        # Handle Dict(image, mission) vs plain Box
        if isinstance(orig_space, gym.spaces.Dict):
            img_space = orig_space["image"]
            self._use_key = "image"
        else:
            img_space = orig_space
            self._use_key = None

        assert isinstance(img_space, gym.spaces.Box), "Expected Box space for image"

        self._img_shape = img_space.shape
        n = int(np.prod(self._img_shape))

        # Normalize to [0,1] for safety
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(n,),
            dtype=np.float32,
        )

    def observation(self, obs):
        if self._use_key is not None:
            img = obs[self._use_key]
        else:
            img = obs

        img = np.asarray(img, dtype=np.float32)
        # Minigrid images are small integers; scale to [0,1]
        img = img / 255.0
        return img.reshape(-1)


def make_single_env(kind: str, seed: int = 0, render_mode=None):
    if kind == "empty":
        env_id = "MiniGrid-Empty-16x16-v0"
    elif kind == "walls":
        env_id = "MiniGrid-FourRooms-v0"
    elif kind == "obstacles":
        env_id = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    else:
        raise ValueError(f"Unknown env kind: {kind}")

    env = gym.make(env_id, render_mode=render_mode)

    # Fully observable symbolic grid
    env = FullyObsWrapper(env)

    # Reward shaping only for harder envs
    if kind in ("walls", "obstacles"):
        env = ShapedRewardWrapper(env, alpha=0.05)

    # Flatten to 1D vector
    env = FlatObsWrapper(env)

    env.reset(seed=seed)
    return env


def _make_env_thunk(kind: str, seed: int):
    def _thunk():
        return make_single_env(kind, seed=seed, render_mode=None)
    return _thunk


def make_vec_env(kind: str, n_envs: int = 4, seed: int = 0):
    set_random_seed(seed)
    env_fns = [_make_env_thunk(kind, seed + i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


def set_global_seeds(seed: int):
    set_random_seed(seed)
