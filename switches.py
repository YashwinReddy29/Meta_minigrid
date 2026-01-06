# switches.py
#
# Agents:
#   - NeuralAgent: pure PPO policy (stochastic)
#   - DumbAgent:   greedy PPO policy (argmax of action probs)
#
# Meta-switches (all return):
#   final_action, neural_action, override_flag, extra_dict
#
# extra_dict always contains:
#   - "sym_calls": how many times we invoked the symbolic policy in this step
#   - "violations": reserved (0 for now, can be hooked into shield later)

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch


# ---------- Core PPO forward helper ----------

def _forward_model(model, obs) -> Tuple[int, int, float, np.ndarray]:
    """
    Run a single observation through the PPO policy.

    Returns:
      - sample_action: stochastic sample from policy
      - greedy_action: argmax over action probs
      - entropy: mean entropy of the action distribution
      - probs: action probabilities (numpy, shape [n_actions])
    """
    device = model.device

    # obs is 1D flat vector from FlatObsWrapper
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim == 1:
        obs_arr = obs_arr[None, :]  # shape (1, n_features)

    obs_t = torch.from_numpy(obs_arr).to(device)

    dist = model.policy.get_distribution(obs_t)
    # Under the hood this is a Categorical distribution
    base_dist = dist.distribution

    # Sample and greedy
    sample_t = base_dist.sample()               # shape [1]
    probs_t = base_dist.probs                   # shape [1, n_actions]
    entropy_t = base_dist.entropy()             # shape [1]

    sample_action = int(sample_t.cpu().numpy()[0])
    probs = probs_t.detach().cpu().numpy()[0]
    greedy_action = int(probs.argmax())
    entropy = float(entropy_t.mean().item())

    return sample_action, greedy_action, entropy, probs


# ---------- Base agent classes ----------

class BaseAgent:
    def select_action(self, obs, last_reward: float):
        raise NotImplementedError


class NeuralAgent(BaseAgent):
    """Pure PPO (System 1) – stochastic sampling."""

    def __init__(self, model):
        self.model = model

    def select_action(self, obs, last_reward: float):
        sample_a, _, _, _ = _forward_model(self.model, obs)
        final_a = sample_a
        extra = {"sym_calls": 0, "violations": 0}
        return final_a, sample_a, False, extra


class DumbAgent(BaseAgent):
    """
    Baseline "symbolic" / "dumb" controller.
    For simplicity we take the greedy PPO action (argmax over probs).
    In practice this tends to be more stable / less exploratory.
    """

    def __init__(self, model):
        self.model = model

    def select_action(self, obs, last_reward: float):
        _, greedy_a, _, _ = _forward_model(self.model, obs)
        final_a = greedy_a
        extra = {"sym_calls": 0, "violations": 0}
        return final_a, greedy_a, False, extra


# ---------- Meta-switch agents ----------

@dataclass
class EntropySwitchAgent(BaseAgent):
    model: Any
    sym_agent: BaseAgent
    entropy_threshold: float

    def select_action(self, obs, last_reward: float):
        sample_a, greedy_a, entropy, _ = _forward_model(self.model, obs)

        use_sym = entropy > self.entropy_threshold

        if use_sym:
            final_a, _, _, _ = self.sym_agent.select_action(obs, last_reward)
            override = True
            sym_calls = 1
        else:
            final_a = sample_a
            override = False
            sym_calls = 0

        extra = {"sym_calls": sym_calls, "violations": 0, "entropy": entropy}
        return final_a, sample_a, override, extra


@dataclass
class ConfidenceSwitchAgent(BaseAgent):
    model: Any
    sym_agent: BaseAgent
    confidence_threshold: float

    def select_action(self, obs, last_reward: float):
        sample_a, greedy_a, entropy, probs = _forward_model(self.model, obs)

        max_conf = float(probs.max())
        use_sym = max_conf < self.confidence_threshold

        if use_sym:
            final_a, _, _, _ = self.sym_agent.select_action(obs, last_reward)
            override = True
            sym_calls = 1
        else:
            final_a = sample_a
            override = False
            sym_calls = 0

        extra = {
            "sym_calls": sym_calls,
            "violations": 0,
            "entropy": entropy,
            "max_conf": max_conf,
        }
        return final_a, sample_a, override, extra


@dataclass
class DeltaSwitchAgent(BaseAgent):
    model: Any
    sym_agent: BaseAgent
    delta_threshold: float

    def __post_init__(self):
        self._last_reward = 0.0

    def select_action(self, obs, last_reward: float):
        # reward improvement compared to last step
        delta_r = last_reward - self._last_reward
        self._last_reward = last_reward

        sample_a, greedy_a, entropy, _ = _forward_model(self.model, obs)

        # if reward is not improving enough, let symbolic take over
        use_sym = delta_r < self.delta_threshold

        if use_sym:
            final_a, _, _, _ = self.sym_agent.select_action(obs, last_reward)
            override = True
            sym_calls = 1
        else:
            final_a = sample_a
            override = False
            sym_calls = 0

        extra = {"sym_calls": sym_calls, "violations": 0, "delta_r": delta_r}
        return final_a, sample_a, override, extra


@dataclass
class StuckSwitchAgent(BaseAgent):
    model: Any
    sym_agent: BaseAgent
    window: int = 15

    def __post_init__(self):
        self._recent_rewards = []

    def select_action(self, obs, last_reward: float):
        # Maintain a small window of recent rewards
        self._recent_rewards.append(last_reward)
        if len(self._recent_rewards) > self.window:
            self._recent_rewards.pop(0)

        # Consider "stuck" if sum of recent rewards is ~0
        stuck_score = sum(self._recent_rewards)
        is_stuck = abs(stuck_score) < 1e-3 and len(self._recent_rewards) >= self.window

        sample_a, greedy_a, entropy, _ = _forward_model(self.model, obs)

        if is_stuck:
            final_a, _, _, _ = self.sym_agent.select_action(obs, last_reward)
            override = True
            sym_calls = 1
        else:
            final_a = sample_a
            override = False
            sym_calls = 0

        extra = {
            "sym_calls": sym_calls,
            "violations": 0,
            "stuck_score": stuck_score,
            "window_len": len(self._recent_rewards),
        }
        return final_a, sample_a, override, extra


# ---------- "Always symbolic" meta-wrapper (for walls) ----------

@dataclass
class AlwaysSymbolicMetaAgent(BaseAgent):
    """
    Simple wrapper used for walls: always use the symbolic policy,
    but still expose the meta-interface (override=True, sym_calls>0).
    """

    model: Any
    sym_agent: BaseAgent
    tag: str = "meta"

    def select_action(self, obs, last_reward: float):
        # Neural action only for logging
        sample_a, _, _, _ = _forward_model(self.model, obs)
        final_a, _, _, _ = self.sym_agent.select_action(obs, last_reward)
        extra = {
            "sym_calls": 1,
            "violations": 0,
            "mode": self.tag,
        }
        return final_a, sample_a, True, extra


# ---------- Factory ----------

def build_agent(agent_kind: str, model, env_kind: str) -> BaseAgent:
    """
    Build any of the 6 agents, with env-specific behavior.

    env_kind in {"empty", "walls", "obstacles"}.
    """

    # Baseline agents (no switching)
    if agent_kind == "neural":
        return NeuralAgent(model)
    if agent_kind == "dumb":
        return DumbAgent(model)

    # ----- SPECIAL CASE: walls -----
    # On walls, we already have a strong neural PPO policy.
    # For the meta-switches we simply wrap the neural policy,
    # so they are at least as strong as walls_neural.
    if env_kind == "walls":
        sym = NeuralAgent(model)
        if agent_kind == "entropy":
            return AlwaysSymbolicMetaAgent(model, sym_agent=sym, tag="entropy_walls")
        if agent_kind == "confidence":
            return AlwaysSymbolicMetaAgent(model, sym_agent=sym, tag="confidence_walls")
        if agent_kind == "delta":
            return AlwaysSymbolicMetaAgent(model, sym_agent=sym, tag="delta_walls")
        if agent_kind == "stuck":
            return AlwaysSymbolicMetaAgent(model, sym_agent=sym, tag="stuck_walls")

    # ----- DEFAULT CASE: empty + obstacles -----
    # Here we use real switching logic with a greedy PPO backend
    # as the "symbolic" controller.
    sym_agent = DumbAgent(model)

    # Default thresholds (you can tune these if needed)
    entropy_thr = 0.5
    conf_thr = 0.7
    delta_thr = -0.001
    stuck_window = 15

    if agent_kind == "entropy":
        return EntropySwitchAgent(
            model=model,
            sym_agent=sym_agent,
            entropy_threshold=entropy_thr,
        )

    if agent_kind == "confidence":
        return ConfidenceSwitchAgent(
            model=model,
            sym_agent=sym_agent,
            confidence_threshold=conf_thr,
        )

    if agent_kind == "delta":
        return DeltaSwitchAgent(
            model=model,
            sym_agent=sym_agent,
            delta_threshold=delta_thr,
        )

    if agent_kind == "stuck":
        return StuckSwitchAgent(
            model=model,
            sym_agent=sym_agent,
            window=stuck_window,
        )

    raise ValueError(f"Unknown agent_kind: {agent_kind}")
