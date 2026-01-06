# visualize_rollouts.py
#
# Create GIF rollouts for a subset of agents on each env.

import os
import imageio.v2 as imageio

from stable_baselines3 import PPO

from envs import make_single_env
from switches import build_agent

OUT_DIR = "runs/rollouts"
MAX_STEPS = 200      # max steps per episode
MAX_TRIES = 20       # max episodes we try to get a success

ENV_KINDS = ["empty", "walls", "obstacles"]
AGENT_KINDS = ["neural", "dumb", "confidence", "entropy", "delta", "stuck"]  # subset for visualization

MODEL_PATHS = {
    "empty": "models/ppo_empty16.zip",
    "walls": "models/ppo_walls16.zip",
    "obstacles": "models/ppo_obstacles16.zip",
}


def render_rollout(env_kind, agent_kind, model, out_dir):
    from envs import make_single_env
    from switches import build_agent

    env = make_single_env(env_kind, seed=42, render_mode="rgb_array")
    agent = build_agent(agent_kind, model, env_kind)

    best_frames = None
    found_success = False

    for attempt in range(MAX_TRIES):
        frames = []
        obs, info = env.reset()
        done = False
        t = 0

        while not done and t < MAX_STEPS:
            frame = env.render()  # rgb_array from minigrid
            frames.append(frame)

            action, _, _, _ = agent.select_action(obs, last_reward=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            t += 1

        # consider success if terminated with positive reward
        if terminated and reward > 0:
            best_frames = frames
            found_success = True
            print(f"[viz] SUCCESS for {env_kind}, {agent_kind} on attempt {attempt+1}, steps={t}")
            break

        # keep last attempt as fallback
        best_frames = frames

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    gif_path = os.path.join(out_dir, f"{env_kind}_{agent_kind}.gif")
    imageio.mimsave(gif_path, best_frames, fps=10)
    print(f"[viz] Saved {gif_path} (success={found_success})")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for env_kind in ENV_KINDS:
        print(f"[viz] env={env_kind}")
        model_path = MODEL_PATHS[env_kind]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model for {env_kind}: {model_path}")
        model = PPO.load(model_path)

        for agent_kind in AGENT_KINDS:
            print(f"[viz] {env_kind}, {agent_kind}")
            render_rollout(env_kind, agent_kind, model, out_dir="runs/rollouts")


if __name__ == "__main__":
    main()
