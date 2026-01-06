# shield.py

from typing import Tuple
from minigrid.core.world_object import Wall


def _forward_pos(raw_env) -> Tuple[int, int]:
    x, y = raw_env.agent_pos
    d = raw_env.agent_dir
    if d == 0:   # right
        return x + 1, y
    if d == 1:   # down
        return x, y + 1
    if d == 2:   # left
        return x - 1, y
    if d == 3:   # up
        return x, y - 1
    return x, y


def _is_hazard(raw_env, x: int, y: int) -> bool:
    # out of bounds = hazard
    if x < 0 or y < 0 or x >= raw_env.width or y >= raw_env.height:
        return True
    cell = raw_env.grid.get(x, y)
    if cell is None:
        return False
    if isinstance(cell, Wall):
        return True
    t = getattr(cell, "type", None)
    if t in ("lava", "door"):
        return True
    if not getattr(cell, "can_overlap", True) and t != "goal":
        return True
    return False


def neural_action_unsafe(raw_env, action: int) -> bool:
    a_enum = raw_env.actions
    if action == a_enum.forward:
        nx, ny = _forward_pos(raw_env)
        return _is_hazard(raw_env, nx, ny)
    return False


def safe_alternative(raw_env, original_action: int) -> int:
    """
    Try left, right, forward in that order, pick first non-hazard move.
    If all are unsafe, keep original.
    """
    a_enum = raw_env.actions
    candidates = [a_enum.left, a_enum.right, a_enum.forward]
    for a in candidates:
        if not neural_action_unsafe(raw_env, a):
            return int(a)
    return int(original_action)
