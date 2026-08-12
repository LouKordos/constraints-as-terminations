from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "create-isaac-lab-env-uv.sh"
ISAACLAB_REVISION = "ddb044eb5b2300792de41e82d53b032f3632b489"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def fake_install_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    command_log = tmp_path / "commands.log"

    write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
set -eu
printf 'UV_PROJECT_ENVIRONMENT=%s uv %s\n' "${UV_PROJECT_ENVIRONMENT:-}" "$*" >> "$COMMAND_LOG"
if [ "${1:-}" = "sync" ]; then
    mkdir -p "${UV_PROJECT_ENVIRONMENT}/bin"
    printf 'deactivate() { :; }\n' > "${UV_PROJECT_ENVIRONMENT}/bin/activate"
    printf '#!/usr/bin/env bash\nexit 0\n' > "${UV_PROJECT_ENVIRONMENT}/bin/python"
    chmod +x "${UV_PROJECT_ENVIRONMENT}/bin/python"
fi
if [ "${1:-}" = "pip" ] && [ "${2:-}" = "check" ]; then
    if [ "${FAKE_UV_PIP_CHECK_MODE:-}" = "known-starlette-conflict" ]; then
        printf 'Found 1 incompatibility\n' >&2
        printf 'The package `fastapi` requires `starlette<0.46.0,>=0.40.0`, but `0.49.1` is installed\n' >&2
        exit 1
    fi
    if [ "${FAKE_UV_PIP_CHECK_MODE:-}" = "unexpected-conflict" ]; then
        printf 'Found 1 incompatibility\n' >&2
        printf 'The package `example` requires `missing`, but it is not installed\n' >&2
        exit 1
    fi
fi
""",
    )
    write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -eu
printf 'git %s\n' "$*" >> "$COMMAND_LOG"
if [ "${1:-}" = "clone" ]; then
    source_arg=$2
    target_arg=${3:-${source_arg##*/}}
    target_arg=${target_arg%.git}
    mkdir -p "$target_arg"
    if [[ "$source_arg" == *IsaacLab.git ]]; then
        mkdir -p "$target_arg/source/isaaclab" \
            "$target_arg/source/isaaclab_assets" \
            "$target_arg/source/isaaclab_tasks" \
            "$target_arg/source/isaaclab_rl"
    else
        mkdir -p "$target_arg/exts/locomposition"
        cp "$PROJECT_TEMPLATE_SOURCE/pyproject.toml" "$target_arg/pyproject.toml"
        cp "$PROJECT_TEMPLATE_SOURCE/uv.lock" "$target_arg/uv.lock"
        cp "$PROJECT_TEMPLATE_SOURCE/train-locomposition.sbatch" "$target_arg/train-locomposition.sbatch"
        cp "$PROJECT_TEMPLATE_SOURCE/exts/locomposition/pyproject.toml" "$target_arg/exts/locomposition/pyproject.toml"
        cp "$PROJECT_TEMPLATE_SOURCE/exts/locomposition/setup.py" "$target_arg/exts/locomposition/setup.py"
    fi
fi
""",
    )

    return {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "COMMAND_LOG": str(command_log),
        "PROJECT_TEMPLATE_SOURCE": str(ROOT),
    }


def run_script(
    tmp_path: Path,
    *arguments: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(fake_install_environment(tmp_path))
    env.update(extra_env or {})
    return subprocess.run(
        [str(SCRIPT), *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def read_commands(tmp_path: Path) -> list[str]:
    return (tmp_path / "commands.log").read_text().splitlines()


def test_setup_script_help_documents_safe_overrides(tmp_path):
    result = run_script(tmp_path, "--help")

    assert result.returncode == 0
    assert "--root PATH" in result.stdout
    assert "--repo-source URL_OR_PATH" in result.stdout


def test_setup_script_rejects_nonempty_target_before_installing(tmp_path):
    install_root = tmp_path / "environments"
    target = install_root / "already-there"
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("do not overwrite")

    result = run_script(
        tmp_path,
        "already-there",
        "--root",
        str(install_root),
        "--repo-source",
        str(ROOT),
    )

    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert (target / "keep.txt").read_text() == "do not overwrite"


def test_setup_script_rejects_unsafe_environment_name(tmp_path):
    result = run_script(
        tmp_path,
        "../outside-root",
        "--root",
        str(tmp_path / "environments"),
    )

    assert result.returncode != 0
    assert "environment name" in result.stderr.lower()


def test_setup_script_rejects_missing_option_value(tmp_path):
    result = run_script(tmp_path, "fresh-env", "--repo-source")

    assert result.returncode != 0
    assert "--repo-source requires" in result.stderr


def test_setup_script_clones_before_frozen_project_sync(tmp_path):
    install_root = tmp_path / "environments"
    result = run_script(
        tmp_path,
        "fresh-env",
        "--root",
        str(install_root),
        "--repo-source",
        "https://github.com/LouKordos/LoComposition.git",
    )

    assert result.returncode == 0, result.stderr
    commands = read_commands(tmp_path)
    clone = (
        "git clone https://github.com/LouKordos/LoComposition.git "
        f"{install_root}/fresh-env/LoComposition"
    )
    venv = install_root / "fresh-env/.venv"
    initial_sync = (
        f"UV_PROJECT_ENVIRONMENT={venv} uv sync --project "
        f"{install_root}/fresh-env/LoComposition --frozen "
        "--no-install-package locomposition"
    )

    assert clone in commands
    assert initial_sync in commands
    assert commands.index(clone) < commands.index(initial_sync)


def test_setup_script_uses_two_lock_boundaries_without_legacy_installs(tmp_path):
    install_root = tmp_path / "environments"
    result = run_script(
        tmp_path,
        "locked-env",
        "--root",
        str(install_root),
        "--repo-source",
        str(ROOT),
    )

    assert result.returncode == 0, result.stderr
    commands = read_commands(tmp_path)
    command_text = "\n".join(commands)
    venv = install_root / "locked-env/.venv"
    repository = install_root / "locked-env/LoComposition"
    target_python = venv / "bin/python"

    assert f"git checkout {ISAACLAB_REVISION}" in commands
    expected_editables = [
        "source/isaaclab",
        "source/isaaclab_assets",
        "source/isaaclab_tasks",
        "source/isaaclab_rl",
    ]
    editable_installs = [
        line for line in commands if " uv pip install " in line and " -e " in line
    ]
    assert len(editable_installs) == 4
    for source in expected_editables:
        assert any(
            f"uv pip install --python {target_python} -e {source}" in line
            for line in editable_installs
        )

    assert (
        f"UV_PROJECT_ENVIRONMENT={venv} uv sync --project {repository} "
        "--frozen --inexact"
    ) in commands
    assert any(f"uv pip check --python {target_python}" in line for line in commands)
    assert "uv init" not in command_text
    assert "uv venv" not in command_text
    assert "--upgrade pip" not in command_text
    assert "-r requirements.txt" not in command_text
    assert "--no-build-isolation" not in command_text
    assert "torch==2.7.0" not in command_text
    assert "isaacsim[all,extscache]==5.1.0" not in command_text


def test_setup_script_generates_l40s_and_2080ti_variants(tmp_path):
    install_root = tmp_path / "environments"
    result = run_script(
        tmp_path,
        "slurm-env",
        "--root",
        str(install_root),
        "--repo-source",
        str(ROOT),
    )

    assert result.returncode == 0, result.stderr
    environment = install_root / "slurm-env"
    l40s = environment / "train-locomposition.sbatch"
    gpu_2080ti = environment / "train-locomposition-2080ti.sbatch"
    assert l40s.is_file()
    assert gpu_2080ti.is_file()

    for job in (l40s, gpu_2080ti):
        syntax = subprocess.run(["bash", "-n", job], text=True, capture_output=True)
        assert syntax.returncode == 0, syntax.stderr
        content = job.read_text()
        assert "__LOCOMPOSITION_ENV_ROOT__" not in content
        assert "__LOCOMPOSITION_ENV_NAME__" not in content
        assert f'ENV_ROOT="{install_root}"' in content
        assert 'ENV_NAME="slurm-env"' in content
        assert "#SBATCH --cpus-per-task=8" in content

    l40s_content = l40s.read_text()
    assert "#SBATCH --job-name=locomposition-training" in l40s_content
    assert "#SBATCH --partition=L40Sday" in l40s_content
    assert "#SBATCH --gres=gpu:L40S:1" in l40s_content
    assert "#SBATCH --array=0-2%3" in l40s_content
    assert "RUNS_PER_NODE=3" in l40s_content

    gpu_2080ti_content = gpu_2080ti.read_text()
    assert "#SBATCH --job-name=locomposition-training-2080ti" in gpu_2080ti_content
    assert "#SBATCH --partition=week" in gpu_2080ti_content
    assert "#SBATCH --gres=gpu:2080ti:1" in gpu_2080ti_content
    assert "#SBATCH --array=0-8%9" in gpu_2080ti_content
    assert "RUNS_PER_NODE=1" in gpu_2080ti_content


def test_setup_script_accepts_only_the_known_upstream_starlette_conflict(tmp_path):
    known = run_script(
        tmp_path / "known",
        "known-conflict",
        "--root",
        str(tmp_path / "known" / "environments"),
        "--repo-source",
        str(ROOT),
        extra_env={"FAKE_UV_PIP_CHECK_MODE": "known-starlette-conflict"},
    )

    assert known.returncode == 0, known.stderr
    assert "known Isaac Sim/Isaac Lab metadata conflict" in known.stderr

    unexpected = run_script(
        tmp_path / "unexpected",
        "unexpected-conflict",
        "--root",
        str(tmp_path / "unexpected" / "environments"),
        "--repo-source",
        str(ROOT),
        extra_env={"FAKE_UV_PIP_CHECK_MODE": "unexpected-conflict"},
    )

    assert unexpected.returncode != 0
    assert "requires `missing`" in unexpected.stderr
