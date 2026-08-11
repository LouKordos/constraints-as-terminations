"""RSL-RL runner configurations for crossed Go2 and ANYmal C tuning."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)


@configclass
class Go2AnymalCTuningPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """ANYmal C PPO parameters for the receiving Go2 embodiment."""

    experiment_name = "go2_anymal_c_tuning_rough"


@configclass
class AnymalCGo2TuningPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Go2 PPO parameters for the receiving ANYmal C embodiment."""

    experiment_name = "anymal_c_go2_tuning_rough"
