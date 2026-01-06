# Cognitive Hybrid Meta-Controller for MiniGrid

This project implements a **cognitive hybrid reinforcement learning system** that combines neural learning (System 1) with symbolic planning and reasoning (System 2) through a meta-controller. The goal is to build an agent that can reliably solve MiniGrid navigation tasks while remaining robust, safe, and interpretable.

Unlike purely neural agents, this system dynamically switches between fast neural policies and deliberate symbolic planning based on uncertainty, confidence, reward dynamics, and behavioral stagnation. The design is inspired by **dual-process cognitive theory**, **hybrid AI**, and **meta-reasoning frameworks**.

---

## 🚀 Key Features

- **System 1 (Neural Agent):**
  - Deep RL agent trained on MiniGrid tasks
  - Learns efficient navigation behavior through interaction

- **System 2 (Symbolic Planner):**
  - BFS / shortest-path planning over the grid
  - Provides safe, goal-directed actions when neural uncertainty is high

- **Cognitive Meta-Controller:**
  - Dynamically switches between System 1 and System 2 using:
    - Entropy-based switching
    - Confidence-based switching
    - Delta-reward (crash detection)
    - Stuck detection

- **Hybrid Training Pipeline:**
  - Behavior cloning warm start
  - Curriculum learning (easy → hard environments)
  - Dense reward shaping for stable learning

- **Environments:**
  - Empty 16×16 MiniGrid
  - Wall-only 16×16 MiniGrid
  - Obstacle-based 16×16 MiniGrid

- **Evaluation & Analysis:**
  - Success rate and episode return
  - Symbolic intervention frequency
  - Safety violations
  - Meta-switch confusion matrix
  - Trajectory and rollout visualization

---

## 🧠 Motivation

Pure reinforcement learning agents often fail in sparse-reward, long-horizon tasks such as grid navigation with obstacles. This project addresses these limitations by integrating **symbolic planning and meta-reasoning**, enabling the agent to reason explicitly when neural decision-making becomes unreliable.

The system follows principles from:
- Hybrid AI and Neuro-Symbolic Reasoning
- Dual-Process Theory (System 1 vs System 2)
- Safe Reinforcement Learning
- Meta-Cognitive Control

---

## 🛠️ Tech Stack

- Python 3
- MiniGrid (Farama Foundation)
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy / Matplotlib
- TensorBoard (optional)

---

## 📁 Project Structure

cognitive_meta_minigrid/
├── agents/ # Neural RL agents
├── envs/ # Custom MiniGrid environments
├── symbolic/ # BFS / symbolic planner
├── switches/ # Meta-controller switching logic
├── train/ # Training scripts
├── eval/ # Evaluation and metrics
├── viz/ # Visualization and rollouts
├── configs/ # Hyperparameters
├── requirements.txt
├── README.md
└── LICENSE
---

## 📊 Outcomes

The hybrid system significantly improves goal-reaching reliability compared to neural-only baselines, especially in obstacle-rich environments. Meta-switching reduces failure modes such as indecision, looping, and unsafe exploration.

---

## 📌 Academic Context

This project was developed as part of a graduate-level course on **Hybrid AI and Cognitive Systems**, and is aligned with current research in meta-reasoning, neuro-symbolic AI, and safe reinforcement learning.

---

## 📜 License

MIT License
