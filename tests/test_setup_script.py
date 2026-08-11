from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "create-isaac-lab-env-uv.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def fake_install_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"

    write_executable(
        fake_bin / "uv",
        """#!/usr/bin/env bash
set -eu
printf 'uv %s\n' "$*" >> "$COMMAND_LOG"
if [ "${1:-}" = "venv" ]; then
    mkdir -p .venv/bin
    printf 'deactivate() { :; }\n' > .venv/bin/activate
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
        mkdir -p "$target_arg/exts/locomposition" "$target_arg/exts/cat_envs"
        : > "$target_arg/requirements.txt"
    fi
fi
""",
    )

    return {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "COMMAND_LOG": str(command_log),
    }


def run_script(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(fake_install_environment(tmp_path))
    return subprocess.run(
        [str(SCRIPT), *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


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


def test_setup_script_clones_and_installs_locomposition(tmp_path):
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
    commands = (tmp_path / "commands.log").read_text()
    assert (
        "git clone https://github.com/LouKordos/LoComposition.git "
        f"{install_root}/fresh-env/LoComposition"
    ) in commands
    assert "uv pip install --no-build-isolation --no-deps -e ./exts/locomposition" in commands
    assert "uv tool update-shell" not in commands
