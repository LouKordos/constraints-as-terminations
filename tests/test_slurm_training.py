from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "train-locomposition.sbatch"
JUSTFILE = ROOT / "justfile"


def render_template(tmp_path: Path) -> Path:
    environment_root = tmp_path / "environments"
    environment_name = "locked-env"
    repository = environment_root / environment_name / "LoComposition"
    virtual_environment = environment_root / environment_name / ".venv"
    repository.mkdir(parents=True)
    (virtual_environment / "bin").mkdir(parents=True)
    (virtual_environment / "bin" / "activate").write_text("")

    rendered = TEMPLATE.read_text()
    rendered = rendered.replace("__LOCOMPOSITION_ENV_ROOT__", str(environment_root))
    rendered = rendered.replace("__LOCOMPOSITION_ENV_NAME__", environment_name)

    destination = tmp_path / "train-locomposition.sbatch"
    destination.write_text(rendered)
    destination.chmod(0o755)
    return destination


def write_fake_just(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "just.log"
    executable = fake_bin / "just"
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$JUST_COMMAND_LOG"
seed=${4:-}
if [ -n "${FAIL_SEED:-}" ] && [ "$seed" = "$FAIL_SEED" ]; then
    exit 17
fi
"""
    )
    executable.chmod(0o755)
    return fake_bin, command_log


def run_rendered_job(
    tmp_path: Path,
    *,
    fail_seed: str | None = None,
    use_api_key: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    job = render_template(tmp_path)
    fake_bin, command_log = write_fake_just(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    if not use_api_key:
        (home / ".netrc").write_text(
            "machine api.wandb.ai login test-user password test-credential\n"
        )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(home),
            "JUST_COMMAND_LOG": str(command_log),
            "SLURM_JOB_ID": "101",
            "SLURM_ARRAY_JOB_ID": "100",
            "SLURM_ARRAY_TASK_ID": "0",
        }
    )
    if fail_seed is not None:
        env["FAIL_SEED"] = fail_seed
    if use_api_key:
        env["WANDB_API_KEY"] = "inherited-test-value"
    else:
        env.pop("WANDB_API_KEY", None)

    result = subprocess.run(
        ["bash", str(job)],
        env=env,
        text=True,
        capture_output=True,
    )
    commands = command_log.read_text().splitlines() if command_log.exists() else []
    return result, commands


def test_l40s_template_has_approved_resources_and_no_secret():
    content = TEMPLATE.read_text()

    assert "#SBATCH --job-name=locomposition-training" in content
    assert "#SBATCH --cpus-per-task=8" in content
    assert "#SBATCH --partition=L40Sday" in content
    assert "#SBATCH --mem-per-cpu=6G" in content
    assert "#SBATCH --gres=gpu:L40S:1" in content
    assert "#SBATCH --array=0-2%3" in content
    assert "RUNS_PER_NODE=3" in content
    assert "NUM_ENVS=7500" in content
    assert 'TASK_NAME="LoComposition-Go2-Rough-Terrain-Joint-State-History-v0"' in content
    assert 'ENV_ROOT="__LOCOMPOSITION_ENV_ROOT__"' in content
    assert 'ENV_NAME="__LOCOMPOSITION_ENV_NAME__"' in content
    assert 'cd "${ENV_ROOT}/${ENV_NAME}/LoComposition"' in content
    assert re.search(r"(?i)(?:export\s+)?WANDB_API_KEY\s*=", content) is None
    assert re.search(r"(?i)\b[0-9a-f]{40}\b", content) is None


def test_l40s_template_is_valid_bash_after_substitution(tmp_path):
    job = render_template(tmp_path)

    result = subprocess.run(["bash", "-n", str(job)], text=True, capture_output=True)

    assert result.returncode == 0, result.stderr


def test_train_recipe_redirects_stderr_into_tee_pipeline():
    content = JUSTFILE.read_text()

    assert " 2>&1 | tee " in content
    assert "| 2>&1 |" not in content


def test_rsl_baseline_recipe_prepares_its_own_log_directory():
    content = JUSTFILE.read_text()
    baseline_recipe = content.split("_train-rsl-baseline", maxsplit=1)[1].split(
        "train-baseline-go2", maxsplit=1
    )[0]

    assert "mkdir -p ./logs/rsl_rl" in baseline_recipe
    assert "mkdir -p ./logs/clean_rl" not in baseline_recipe


def test_array_task_zero_launches_three_expected_runs(tmp_path):
    result, commands = run_rendered_job(tmp_path)

    assert result.returncode == 0, result.stderr
    assert sorted(commands) == [
        "train 7500 LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 46",
        "train 7500 LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 47",
        "train 7500 LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 48",
    ]


def test_inherited_wandb_api_key_is_accepted_without_netrc(tmp_path):
    result, commands = run_rendered_job(tmp_path, use_api_key=True)

    assert result.returncode == 0, result.stderr
    assert len(commands) == 3
    assert "inherited-test-value" not in result.stdout
    assert "inherited-test-value" not in result.stderr


def test_one_failed_child_makes_job_fail_after_all_children_finish(tmp_path):
    result, commands = run_rendered_job(tmp_path, fail_seed="47")

    assert result.returncode != 0
    assert len(commands) == 3
    assert "at least one training run failed" in result.stdout
