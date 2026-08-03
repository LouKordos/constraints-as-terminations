from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _dry_run(*arguments: str) -> str:
    result = subprocess.run(
        ["just", "--dry-run", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    "recipe,task,default_project",
    [
        (
            "train-baseline-go2-anymal-c-tuning",
            "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0",
            "baseline_go2_anymal_c_rewards_action_scale_ppo",
        ),
        (
            "train-baseline-anymal-c-go2-tuning",
            "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0",
            "baseline_anymal_c_go2_rewards_action_scale_ppo",
        ),
    ],
)
def test_cross_tuning_recipe_defaults(recipe, task, default_project):
    public_expansion = _dry_run(recipe)
    trainer_expansion = _dry_run(
        "_train-rsl-baseline",
        task,
        "7500",
        "46",
        "30000",
        default_project,
    )

    assert f"just _train-rsl-baseline {task} 7500 46 30000 {default_project}" in public_expansion
    assert f"--task={task}" in trainer_expansion
    assert "--num_envs=7500" in trainer_expansion
    assert "--seed=46" in trainer_expansion
    assert "--max_iterations=30000" in trainer_expansion
    assert "--logger=wandb" in trainer_expansion
    assert f"--log_project_name={default_project}" in trainer_expansion


@pytest.mark.parametrize(
    "recipe,task",
    [
        (
            "train-baseline-go2-anymal-c-tuning",
            "Baseline-Go2-Anymal-C-Tuning-Rough-Terrain-v0",
        ),
        (
            "train-baseline-anymal-c-go2-tuning",
            "Baseline-Anymal-C-Go2-Tuning-Rough-Terrain-v0",
        ),
    ],
)
def test_cross_tuning_recipe_forwards_overrides_and_flags(recipe, task):
    arguments = (
        "64",
        "47",
        "2",
        "custom_cross_project",
        "--run_name=cross_smoke",
    )
    public_expansion = _dry_run(recipe, *arguments)
    trainer_expansion = _dry_run("_train-rsl-baseline", task, *arguments)

    assert (
        f"just _train-rsl-baseline {task} 64 47 2 custom_cross_project "
        "--run_name=cross_smoke"
    ) in public_expansion
    assert f"--task={task}" in trainer_expansion
    assert "--num_envs=64" in trainer_expansion
    assert "--seed=47" in trainer_expansion
    assert "--max_iterations=2" in trainer_expansion
    assert "--log_project_name=custom_cross_project" in trainer_expansion
    assert "--run_name=cross_smoke" in trainer_expansion
