# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Curriculum function which gets called by Isaac Lab at certain interval. The intended effect is to slowly ramp up the maximum allowed termination probability
# to prevent overwhelming the agent/policy with strict constraints early on. As the policy learns, the constraints are gradually enforced more and more by increasing
# the maximum termination probability that a constraint violation can evoke.
def modify_constraint_p(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, num_steps: int, init_max_p: float,):
    # Set to 0 if it's the first iteration
    if not hasattr(env.curriculum_manager, "constraints_curriculum"):
        env.curriculum_manager.constraints_curriculum = 0

    # Expected step per function call, i.e. the max_p increment per function call.
    current_curriculum_progress_step = 1.0 / num_steps

    env.curriculum_manager.constraints_curriculum = min(env.curriculum_manager.constraints_curriculum + current_curriculum_progress_step, 1.0)

    # Linearly interpolate the expected time for episode end: soft_p is the maximum
    # termination probability so it is an image of the expected time of death.
    T_start = 20
    T_end = 1 / init_max_p
    interpolated_new_termination_time = T_start + env.curriculum_manager.constraints_curriculum * (T_end - T_start)
    init_max_p = 1 / (interpolated_new_termination_time)

    # obtain term settings
    term_cfg = env.constraint_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.max_p = init_max_p
    env.constraint_manager.set_term_cfg(term_name, term_cfg)

    return init_max_p

def update_reward_weight_linear(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, num_steps: int, start_at_step: int, start_weight: float, end_weight: float):
    progress = min(max(0, (env.common_step_counter - start_at_step)) / num_steps, 1.0)
    new_weight = start_weight + (end_weight - start_weight) * progress
    
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)

    return {"weight": new_weight}