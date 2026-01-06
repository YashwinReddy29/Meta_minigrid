from dataclasses import dataclass

@dataclass
class PPOConfig:
    n_envs: int = 4
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 2048
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2

# per-env total timesteps (buff walls + obstacles)
ENV_TIMESTEPS = {
    "empty": 200_000,      # stays the same
    "walls": 1_500_000,    # much more training for FourRooms
    "obstacles": 1_200_000,
}


ppo_cfg = PPOConfig()
