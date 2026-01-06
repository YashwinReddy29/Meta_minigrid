import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import ppo_cfg, ENV_TIMESTEPS
from envs import make_vec_env, set_global_seeds


def build_model(env_kind: str, env, seed: int):
    """
    Build a PPO model with simple curriculum:

      - empty:   train from scratch
      - walls:   if ppo_walls16.zip exists -> continue training it
                 elif ppo_empty16.zip exists -> start from empty policy
                 else -> train from scratch
      - obstacles: if ppo_obstacles16.zip exists -> continue
                   elif ppo_walls16.zip exists -> start from walls policy
                   else -> from scratch
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    target_path = os.path.join(models_dir, f"ppo_{env_kind}16.zip")

    # 1) If we already have a trained model for this env, keep training it
    if os.path.exists(target_path):
        print(f"[train_ppo] loading existing model for {env_kind} from {target_path}")
        model = PPO.load(target_path)
        model.set_env(env)
        return model

    # 2) Curriculum: walls from empty, obstacles from walls, if available
    if env_kind == "walls":
        src_path = os.path.join(models_dir, "ppo_empty16.zip")
        if os.path.exists(src_path):
            print(f"[train_ppo] initializing walls from empty policy: {src_path}")
            model = PPO.load(src_path)
            model.set_env(env)
            return model

    if env_kind == "obstacles":
        src_path = os.path.join(models_dir, "ppo_walls16.zip")
        if os.path.exists(src_path):
            print(f"[train_ppo] initializing obstacles from walls policy: {src_path}")
            model = PPO.load(src_path)
            model.set_env(env)
            return model

    # 3) Fallback: fresh model
    print(f"[train_ppo] training {env_kind} from scratch")
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
        seed=seed,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["empty", "walls", "obstacles"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env_kind = args.env
    seed = args.seed

    set_global_seeds(seed)
    env = make_vec_env(env_kind, n_envs=ppo_cfg.n_envs, seed=seed)

    model = build_model(env_kind, env, seed)

    total_ts = ENV_TIMESTEPS[env_kind]
    print(f"[train_ppo] env={env_kind}, total_timesteps={total_ts}")

    ckpt_cb = CheckpointCallback(
        save_freq=total_ts // 5,
        save_path="models",
        name_prefix=f"ppo_{env_kind}16_ckpt",
    )

    model.learn(total_timesteps=total_ts, callback=ckpt_cb)
    model.save(os.path.join("models", f"ppo_{env_kind}16.zip"))


if __name__ == "__main__":
    main()
