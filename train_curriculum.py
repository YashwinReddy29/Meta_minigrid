# train_curriculum.py
#
# Sequentially train one PPO policy on:
#   1) empty 16x16
#   2) FourRooms (walls)
#   3) Dynamic Obstacles 16x16
#
# So the final model has seen all three env families.

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import ppo_cfg
from envs import make_vec_env, set_global_seeds


def train_on_stage(model, env_kind: str, timesteps: int, seed: int, stage_name: str):
    """
    Train PPO on a single env kind for `timesteps`.
    If model is None, create a new PPO; otherwise reuse it and just swap env.
    """
    print(f"[curriculum] Stage={stage_name}, env={env_kind}, timesteps={timesteps}")
    env = make_vec_env(env_kind, n_envs=ppo_cfg.n_envs, seed=seed)

    if model is None:
        model = PPO(
            policy=ppo_cfg.policy,
            env=env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            ent_coef=ppo_cfg.ent_coef,
            clip_range=ppo_cfg.clip_range,
            verbose=1,
            tensorboard_log=None,
            seed=seed,
        )
    else:
        # Reuse the same network, just swap environments.
        model.set_env(env)

    ckpt_cb = CheckpointCallback(
        save_freq=max(10_000 // ppo_cfg.n_envs, 1),
        save_path="runs",
        name_prefix=f"ppo_curr_{stage_name}",
    )

    model.learn(
        total_timesteps=timesteps,
        callback=ckpt_cb,
        reset_num_timesteps=False,  # keep counting across stages
    )

    env.close()
    return model


def main():
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    seed = 42
    set_global_seeds(seed)

    # Curriculum schedule: (env_kind, timesteps, label)
    stages = [
        ("empty", 60_000, "empty16"),
        ("walls", 60_000, "walls_fourrooms"),
        ("obstacles", 60_000, "dyn_obs16"),
    ]

    model = None
    for env_kind, t_steps, label in stages:
        model = train_on_stage(model, env_kind, t_steps, seed, label)

    model.save("models/ppo_curriculum16")
    print("[curriculum] Saved model to models/ppo_curriculum16")


if __name__ == "__main__":
    main()
