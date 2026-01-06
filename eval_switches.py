# eval_switches.py
#
# Evaluate 6 agents (neural, dumb, 4 meta-switches)
# on 3 envs (empty, walls, obstacles).
# Save results to CSV and make comparison plots.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from envs import make_single_env
from switches import build_agent

RESULTS_DIR = "runs"
N_EPISODES = 30
MAX_STEPS = 200  # safety cap per episode


ENV_KINDS = ["empty", "walls", "obstacles"]
AGENT_KINDS = ["neural", "dumb", "entropy", "confidence", "delta", "stuck"]

MODEL_PATHS = {
    "empty": "models/ppo_empty16.zip",
    "walls": "models/ppo_walls16.zip",
    "obstacles": "models/ppo_obstacles16.zip",
}


def run_eval(env_kind: str, agent_kind: str, model):
    env = make_single_env(env_kind, seed=123, render_mode=None)
    agent = build_agent(agent_kind, model, env_kind)

    successes = 0
    violations = 0
    total_reward = 0.0
    total_steps = 0
    total_sym_calls = 0
    episodes = 0

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        last_reward = None
        steps = 0

        while not (done or truncated) and steps < MAX_STEPS:
            action, neural_action, override, extra = agent.select_action(obs, last_reward)
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += reward
            steps += 1
            last_reward = reward
            total_sym_calls += extra.get("sym_calls", 0)

        total_reward += ep_rew
        total_steps += steps
        episodes += 1

        # simple success/violation logic:
        #   success: positive final reward
        #   violation: negative final reward
        if ep_rew > 0:
            successes += 1
        elif ep_rew < 0:
            violations += 1

    env.close()

    return {
        "env": env_kind,
        "agent": agent_kind,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "violation_rate": violations / episodes,
        "avg_return": total_reward / episodes,
        "avg_steps": total_steps / episodes,
        "avg_sym_calls": total_sym_calls / episodes,
    }


def make_plots(df: pd.DataFrame):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for metric in ["success_rate", "violation_rate", "avg_return", "avg_steps", "avg_sym_calls"]:
        plt.figure()
        for env_kind in ENV_KINDS:
            sub = df[df["env"] == env_kind]
            plt.plot(sub["agent"], sub[metric], marker="o", label=env_kind)
        plt.xlabel("Agent")
        plt.ylabel(metric)
        plt.title(f"{metric} per agent and env")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}_by_agent_env.png"))
        plt.close()


def main():
    from stable_baselines3 import PPO
    import pandas as pd

    ENV_KINDS = ["empty", "walls", "obstacles"]
    AGENT_KINDS = ["neural", "dumb", "entropy", "confidence", "delta", "stuck"]
    N_EPISODES = 50

    records = []

    for env_kind in ENV_KINDS:
        print(f"[eval] env={env_kind}")
        env = make_single_env(env_kind, seed=42, render_mode=None)

        model_path = f"models/ppo_{env_kind}16.zip"
        model = PPO.load(model_path)

        for agent_kind in AGENT_KINDS:
            print(f"[eval] env={env_kind}, agent={agent_kind}")
            agent = build_agent(agent_kind, model, env_kind)

            for ep in range(N_EPISODES):
                obs, info = env.reset()
                done = False
                ep_return = 0.0
                steps = 0
                last_reward = 0.0
                sym_calls = 0
                violations = 0

                while not done:
                    final_a, neural_a, override, extra = agent.select_action(obs, last_reward)
                    obs, reward, terminated, truncated, info = env.step(final_a)
                    done = terminated or truncated

                    ep_return += reward
                    steps += 1
                    last_reward = reward

                    # meta switch stats (use .get so it works for all agents)
                    sym_calls += extra.get("sym_calls", 0)
                    violations += extra.get("violations", 0)

                records.append(
                    {
                        "env": env_kind,
                        "agent": agent_kind,
                        "episode": ep,
                        "return": ep_return,
                        "steps": steps,
                        "sym_calls": sym_calls,
                        "violations": violations,
                        "success": float(ep_return > 0),
                    }
                )

    df = pd.DataFrame(records)
    os.makedirs("runs", exist_ok=True)
    out_path = os.path.join("runs", "results_switches.csv")
    df.to_csv(out_path, index=False)
    print(f"[eval] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
