# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

from . import agents
from cat_envs.tasks.utils.cat.cat_env import CaTEnv


##
# Register Gym environments.
##

_GO2_RSL_RL_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
    "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
)
_ANYMAL_C_RSL_RL_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents."
    "rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg"
)
_SPOT_RSL_RL_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents."
    "rsl_rl_ppo_cfg:SpotFlatPPORunnerCfg"
)
_GO2_ANYMAL_C_TUNING_RSL_RL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AnymalCTuningPPORunnerCfg"
)
_ANYMAL_C_GO2_TUNING_RSL_RL_CFG = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCGo2TuningPPORunnerCfg"
)


gym.register(
    id="Baseline-Go2-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_go2_rough_env_cfg:BaselineGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": _GO2_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Go2-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_go2_rough_env_cfg:BaselineGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _GO2_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Anymal-C-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_anymal_c_rough_env_cfg:BaselineAnymalCRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": _ANYMAL_C_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Anymal-C-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_anymal_c_rough_env_cfg:BaselineAnymalCRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _ANYMAL_C_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineGo2AnymalCTuningRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": _GO2_ANYMAL_C_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineGo2AnymalCTuningRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _GO2_ANYMAL_C_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineAnymalCGo2TuningRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": _ANYMAL_C_GO2_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_cross_tuning_env_cfg:"
            "BaselineAnymalCGo2TuningRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _ANYMAL_C_GO2_TUNING_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Spot-Rough-Terrain-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_spot_rough_env_cfg:BaselineSpotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": _SPOT_RSL_RL_CFG,
    },
)

gym.register(
    id="Baseline-Spot-Rough-Terrain-Play-v0",
    entry_point=ManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.baseline_spot_rough_env_cfg:BaselineSpotRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": _SPOT_RSL_RL_CFG,
    },
)

gym.register(
    id="Isaac-Velocity-CaT-Flat-Solo12-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-CaT-Rectangular-Stairs-Solo12-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_rectangular_stairs_env_cfg:Solo12RectangularStairsEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Go2-Rough-Terrain-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_go2_rough_terrain_env_cfg:Go2RoughTerrainEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Anymal-C-Rough-Terrain-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_anymal_c_rough_terrain_env_cfg:AnymalCRoughTerrainEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Spot-Rough-Terrain-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_spot_rough_terrain_env_cfg:SpotRoughTerrainEnvCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Go2-Rough-Terrain-Joint-State-History-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_go2_rough_terrain_env_cfg:Go2RoughTerrainEnvCfgJointStateHistory",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Go2-Rough-Terrain-Full-State-History-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_go2_rough_terrain_env_cfg:Go2RoughTerrainEnvCfgFullStateHistory",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-CaT-Flat-Solo12-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_flat_env_cfg:Solo12FlatEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-CaT-Rectangular-Stairs-Solo12-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_rectangular_stairs_env_cfg:Solo12RectangularStairsEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Go2-Rough-Terrain-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_go2_rough_terrain_env_cfg:Go2RoughTerrainEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Anymal-C-Rough-Terrain-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_anymal_c_rough_terrain_env_cfg:AnymalCRoughTerrainEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Spot-Rough-Terrain-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_spot_rough_terrain_env_cfg:SpotRoughTerrainEnvCfg_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Go2-Rough-Terrain-Joint-State-History-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_go2_rough_terrain_env_cfg:Go2RoughTerrainEnvCfgJointStateHistory_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)

gym.register(
    id="CaT-Go2-Rough-Terrain-Full-State-History-Play-v0",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_go2_rough_terrain_env_cfg:Go2RoughTerrainEnvCfgFullStateHistory_PLAY",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Solo12FlatPPORunnerCfg",
    },
)
