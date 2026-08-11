# LoComposition Repository Rebrand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebrand the repository and its project-owned software as LoComposition while retaining CaT terminology for the underlying constraint mechanism and preserving existing Python, Gymnasium, checkpoint, and primary ROS launch workflows wherever compatibility is technically honest.

**Architecture:** The canonical Python and ROS namespaces move to `locomposition*`. A small Python import alias and explicit legacy Gym registrations preserve old research workflows without duplicating implementations. ROS launch wrappers preserve command-level compatibility, while the project-owned message type deliberately moves to `locomposition_perception_msgs` and is documented as a source-level migration because ROS type names are wire identities.

**Tech Stack:** Python 3.11, Isaac Sim 5.1.0, Isaac Lab at commit `ddb044eb5b2300792de41e82d53b032f3632b489`, PyTorch 2.7.0/CUDA 12.8 wheels, Gymnasium, CleanRL PPO, ROS 2, CMake, ament, Docker Compose, pytest, Markdown, and Poppler asset conversion.

## Global Constraints

- The future repository and project name is exactly `LoComposition`; Python and ROS package names use lowercase `locomposition`.
- Do not change reward, constraint, observation, action, terrain, curriculum, domain-randomization, PPO, checkpoint, or evaluation behavior as part of the rename.
- Keep `CaT`, `CaTEnv`, `ConstraintManager`, `tasks/utils/cat`, original Solo12 CaT tasks, provenance comments, and CaT citations when they refer to the actual mechanism or upstream work.
- New Go2, ANYmal C, and Spot tasks use `LoComposition-*`; all existing `CaT-*` task IDs remain registered against equivalent entry points.
- Existing `cat_envs` imports remain supported without loading duplicate class objects.
- Rename the project-owned ROS message package to `locomposition_perception_msgs`; do not claim that the old and new ROS message types are wire-compatible.
- Keep the README focused on the released software and route long scientific or deployment explanations to the paper, project page, blog, or focused docs.
- Cite LoComposition and explicitly cite Chane-Sane et al. for Constraints as Terminations.
- Describe ANYmal C and Spot as separately trained formulation-transfer results. State that action/actuator settings follow the embodiment and that disturbances, mass randomization, and energy coefficients are scaled deterministically by mass rather than retuned through a new search.
- The owner performs the GitHub repository rename only after this branch is complete.
- Ignore the known unrestricted pytest collection errors for absent `ament_copyright`, `ament_flake8`, and `ament_pep257`; do not ignore focused tests or real workflow failures.
- The paper source at `/home/kordoslo/Downloads/EfficientLocomotion_ConstraintRL_Loukas` is authoritative for title, authors, figures, and paper-result wording.

---

## File Structure

### Canonical Python extension

- `exts/locomposition/locomposition/`: canonical implementation, moved from `exts/cat_envs/cat_envs/`.
- `exts/locomposition/cat_envs/__init__.py`: legacy namespace loader forwarding `cat_envs[.*]` imports to canonical modules.
- `exts/locomposition/config/extension.toml`: LoComposition extension metadata.
- `exts/locomposition/setup.py`: installs both the canonical package and the compatibility package.
- `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/locomposition_*_rough_terrain_env_cfg.py`: canonical project configurations.
- `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_*_rough_terrain_env_cfg.py`: three small legacy module forwarders for old module paths; `cat_flat_env_cfg.py` remains a real upstream CaT example.

### Tests and verification helpers

- `tests/test_python_rebrand_contract.py`: metadata, source-layout, and compatibility-loader contract.
- `tests/test_task_naming.py`: pure task-name classification and canonicalization tests.
- `tests/test_setup_script.py`: setup-script argument and path tests without downloading Isaac Sim.
- `tests/test_ros_rebrand_contract.py`: ROS package, dependency, and wrapper-layout checks.
- `tests/test_readme_contract.py`: README structure, relative-link, asset, and attribution checks.
- `scripts/task_naming.py`: pure old/new task-name helpers shared by evaluation code.
- `scripts/verify_task_aliases.py`: Isaac-aware task registry/configuration verification launched after `AppLauncher`.

### ROS 2 workspace

- `sim2real/ros2_ws/src/locomposition_controller/`: canonical C++ controller and map-processing nodes.
- `sim2real/ros2_ws/src/locomposition_bringup/`: canonical high-level launch package.
- `sim2real/ros2_ws/src/locomposition_state_estimation/`: canonical odometry/LiDAR launch package.
- `sim2real/ros2_ws/src/locomposition_perception_msgs/`: canonical `ProcessedElevationMap` message.
- `sim2real/ros2_ws/src/cat_controller/`, `cat_bringup/`, and `cat_state_estimation/`: launch-only compatibility packages with no duplicated controller implementation.

### Documentation and media

- `README.md`: concise repository front page.
- `CITATION.cff`: machine-readable LoComposition citation metadata.
- `docs/sim2real.md`: detailed hardware/deployment guide moved from the old README.
- `docs/migration.md`: old/new names and final GitHub rename checklist.
- `assets/locomposition-overview.png`, `assets/cot-and-contact-patterns.png`, `assets/sim2real-overview.png`, `assets/terrain-contact-adaptation.png`, and `assets/swing-height-adaptation.png`: web exports from the supplied paper source.
- `assets/demos/go2-hardware.gif`, `assets/demos/anymal-c.gif`, and `assets/demos/spot.gif`: visibly labeled media placeholders that are not presented as experimental evidence.

---

### Task 1: Migrate the Python package and preserve legacy imports

**Files:**
- Create: `tests/test_python_rebrand_contract.py`
- Move: `exts/cat_envs/` to `exts/locomposition/`
- Move: `exts/locomposition/cat_envs/` to `exts/locomposition/locomposition/`
- Create: `exts/locomposition/cat_envs/__init__.py`
- Create: `exts/locomposition/locomposition/_legacy_namespace.py`
- Modify: `exts/locomposition/setup.py`
- Modify: `exts/locomposition/config/extension.toml`
- Modify: `pyproject.toml`
- Modify: `.fdignore`
- Modify: `exts/locomposition/locomposition/assets/odri.py`
- Modify: all canonical Python imports under `exts/locomposition/locomposition/`
- Modify: `scripts/clean_rl/train.py`
- Modify: `scripts/clean_rl/play.py`
- Modify: `scripts/list_envs.py`
- Modify: `scripts/trace_model_checkpoint.py`
- Modify: `scripts/train_rsl_rl.py`

**Interfaces:**
- Produces: canonical import root `locomposition`.
- Produces: legacy import root `cat_envs`, with `cat_envs.<suffix>` resolving to the same module object as `locomposition.<suffix>`.
- Preserves: `CaT`, `CaTEnv`, `ConstraintManager`, and the `tasks.utils.cat` module path below either root.

- [ ] **Step 1: Write the static package-contract tests**

```python
# tests/test_python_rebrand_contract.py
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_primary_python_package_is_locomposition():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "locomposition"
    assert (ROOT / "exts/locomposition/locomposition/__init__.py").is_file()


def test_legacy_namespace_is_an_explicit_forwarder():
    shim = ROOT / "exts/locomposition/cat_envs/__init__.py"
    text = shim.read_text()
    assert "install_legacy_namespace" in text
    assert 'canonical_root="locomposition"' in text
    assert 'legacy_root="cat_envs"' in text


def test_cat_algorithm_directory_remains_named_cat():
    assert (
        ROOT
        / "exts/locomposition/locomposition/tasks/utils/cat/constraint_manager.py"
    ).is_file()
```

- [ ] **Step 2: Run the tests and confirm the pre-rename failure**

Run: `pytest -q tests/test_python_rebrand_contract.py`

Expected: FAIL because `pyproject.toml` still names `cat-isaac-lab-env` and `exts/locomposition` does not exist.

- [ ] **Step 3: Move the extension and canonical package**

Use Git-aware moves for the directory trees, then update repository-relative asset paths from `exts/cat_envs/cat_envs` to `exts/locomposition/locomposition`. Preserve binary robot assets unchanged.

Expected canonical layout:

```text
exts/locomposition/
├── config/extension.toml
├── locomposition/
│   ├── assets/
│   └── tasks/
└── setup.py
```

- [ ] **Step 4: Implement one canonical-to-legacy module alias loader**

```python
# exts/locomposition/locomposition/_legacy_namespace.py
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


class _LegacyAliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_name: str):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        return importlib.import_module(self.canonical_name)

    def exec_module(self, module):
        return None


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, canonical_root: str, legacy_root: str):
        self.canonical_root = canonical_root
        self.legacy_root = legacy_root

    def find_spec(self, fullname, path=None, target=None):
        prefix = f"{self.legacy_root}."
        if not fullname.startswith(prefix):
            return None
        canonical_name = f"{self.canonical_root}.{fullname[len(prefix):]}"
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _LegacyAliasLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


def install_legacy_namespace(*, canonical_root: str, legacy_root: str) -> None:
    if not any(
        isinstance(finder, _LegacyAliasFinder)
        and finder.canonical_root == canonical_root
        and finder.legacy_root == legacy_root
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _LegacyAliasFinder(canonical_root, legacy_root))
    sys.modules[legacy_root] = importlib.import_module(canonical_root)
```

```python
# exts/locomposition/cat_envs/__init__.py
"""Compatibility namespace for the former project package name."""

from locomposition._legacy_namespace import install_legacy_namespace

install_legacy_namespace(canonical_root="locomposition", legacy_root="cat_envs")
```

Exercise this loader with both import orders. If Python's import machinery mutates the canonical module metadata for an alias import, fix the loader rather than weakening the identity requirement.

- [ ] **Step 5: Update package metadata and discovery**

Set the root distribution name and description:

```toml
[project]
name = "locomposition"
description = "LoComposition locomotion learning environments and tools"
```

Update `exts/locomposition/setup.py` to install both trees:

```python
from setuptools import find_packages, setup

setup(
    name="locomposition",
    packages=find_packages(include=("locomposition", "locomposition.*", "cat_envs")),
    # retain the existing version, dependencies, classifiers, and package data
)
```

Set the extension title to `LoComposition`, repository to
`https://github.com/LouKordos/LoComposition`, module name to `locomposition`,
and author/maintainer to the LoComposition project rather than the Isaac Lab
template defaults. Set `known_firstparty = ["locomposition", "cat_envs"]` in
the root formatter configuration.

- [ ] **Step 6: Change project imports to the canonical namespace**

Replace project-owned imports such as:

```python
from cat_envs.tasks.utils.cleanrl.ppo import PPO
```

with:

```python
from locomposition.tasks.utils.cleanrl.ppo import PPO
```

Do not rename local variables that use `cat_` to mean “ported from CaT” unless
they describe the whole LoComposition project. Do not touch `torch.cat`.

- [ ] **Step 7: Run static and runtime import checks**

Run: `pytest -q tests/test_python_rebrand_contract.py`

Expected: PASS.

Run after launching the installed Isaac environment:

```bash
python -c 'import locomposition, cat_envs; from locomposition.tasks.utils.cat.cat_env import CaTEnv as New; from cat_envs.tasks.utils.cat.cat_env import CaTEnv as Old; assert New is Old'
```

Expected: exit 0 with no warning and no duplicate-class assertion failure.

- [ ] **Step 8: Commit the Python namespace migration**

```bash
git add .fdignore pyproject.toml exts/locomposition scripts tests/test_python_rebrand_contract.py
git commit -m "Rename Python package to LoComposition"
```

---

### Task 2: Add canonical LoComposition tasks and retain CaT task aliases

**Files:**
- Create: `tests/test_task_naming.py`
- Create: `scripts/task_naming.py`
- Create: `scripts/verify_task_aliases.py`
- Move: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py` to `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/locomposition_go2_rough_terrain_env_cfg.py`
- Move: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py` to `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/locomposition_anymal_c_rough_terrain_env_cfg.py`
- Move: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py` to `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/locomposition_spot_rough_terrain_env_cfg.py`
- Create: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_go2_rough_terrain_env_cfg.py`
- Create: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_anymal_c_rough_terrain_env_cfg.py`
- Create: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/cat_spot_rough_terrain_env_cfg.py`
- Modify: `exts/locomposition/locomposition/tasks/locomotion/velocity/config/solo12/__init__.py`
- Modify: all baseline/config imports of the moved modules
- Modify: `scripts/eval.py`
- Modify: `justfile`
- Modify: `scripts/runpod-setup.sh`
- Modify: `scripts/clean_rl/train.py`
- Modify: `scripts/clean_rl/play.py`

**Interfaces:**
- Produces: `canonical_task_id(task_id: str) -> str` and task-family predicates that recognize old and new names.
- Produces: paired Gymnasium registrations for every project train/play environment.
- Preserves: every existing `CaT-Go2-*`, `CaT-Anymal-C-*`, and `CaT-Spot-*` ID.

- [ ] **Step 1: Write pure task-name tests**

```python
# tests/test_task_naming.py
from scripts.task_naming import canonical_task_id, is_locomposition_go2_task


def test_legacy_go2_task_canonicalizes():
    assert canonical_task_id("CaT-Go2-Rough-Terrain-Play-v0") == (
        "LoComposition-Go2-Rough-Terrain-Play-v0"
    )


def test_new_and_legacy_go2_names_are_recognized():
    assert is_locomposition_go2_task("LoComposition-Go2-Rough-Terrain-v0")
    assert is_locomposition_go2_task("CaT-Go2-Rough-Terrain-v0")


def test_upstream_cat_example_is_not_rebranded():
    task = "Isaac-Velocity-CaT-Flat-Solo12-v0"
    assert canonical_task_id(task) == task
```

- [ ] **Step 2: Run the tests and confirm the missing-helper failure**

Run: `pytest -q tests/test_task_naming.py`

Expected: FAIL with `ModuleNotFoundError: scripts.task_naming`.

- [ ] **Step 3: Implement the pure task-name mapping**

```python
# scripts/task_naming.py
LEGACY_PREFIX = "CaT-"
CANONICAL_PREFIX = "LoComposition-"


def canonical_task_id(task_id: str) -> str:
    project_families = ("Go2-", "Anymal-C-", "Spot-")
    if any(
        task_id.startswith(LEGACY_PREFIX + family)
        for family in project_families
    ):
        return CANONICAL_PREFIX + task_id.removeprefix(LEGACY_PREFIX)
    return task_id


def is_locomposition_go2_task(task_id: str) -> bool:
    normalized = canonical_task_id(task_id).lower().replace("_", "-")
    return normalized.startswith("locomposition-go2-")
```

Use an explicit family tuple; do not turn every historical task containing
`CaT` into LoComposition.

- [ ] **Step 4: Move the project configuration modules and retain old module paths**

The canonical Go2 module keeps the real implementation and the existing
descriptive class names such as `Go2RoughTerrainEnvCfg`; those names never
claimed that the full project was CaT and changing them would add needless API
breakage. Only the project-branded module path changes.

Each old module path is a small forwarder:

```python
"""Compatibility imports for the former CaT-branded project module."""

from .locomposition_go2_rough_terrain_env_cfg import *  # noqa: F401,F403
```

Apply the same pattern to ANYmal C and Spot. Leave `cat_flat_env_cfg.py` as the
real original CaT example.

- [ ] **Step 5: Register canonical and legacy IDs from one helper**

```python
def _register_project_task(*, task_id, legacy_task_id, entry_point, kwargs):
    for registered_id in (task_id, legacy_task_id):
        gym.register(
            id=registered_id,
            entry_point=entry_point,
            disable_env_checker=True,
            kwargs=dict(kwargs),
        )
```

Use the canonical module/class in `kwargs` for both registrations. Do not
duplicate configuration dictionaries by hand. Keep baseline and original
Solo12 registrations unchanged.

- [ ] **Step 6: Make scripts prefer new IDs while accepting old IDs**

Import the pure helpers in `scripts/eval.py`. Change observation-dimension
autoselection to the new Go2 play IDs. Replace ad hoc checks for `cat-go2` with
`is_locomposition_go2_task()`. Change error messages from “custom CaT task
package” to “LoComposition task package.” Import `locomposition` in training,
play, listing, tracing, and RSL-RL scripts.

Change the default `just train` task to
`LoComposition-Go2-Rough-Terrain-Joint-State-History-v0`. Update RunPod clone,
install, and example commands to the future repository/package names.

- [ ] **Step 7: Add Isaac-aware registry verification**

```python
# scripts/verify_task_aliases.py
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
try:
    import gymnasium as gym
    import locomposition.tasks.locomotion.velocity.config.solo12  # noqa: F401

    pairs = {
        "LoComposition-Go2-Rough-Terrain-v0": "CaT-Go2-Rough-Terrain-v0",
        "LoComposition-Go2-Rough-Terrain-Play-v0": "CaT-Go2-Rough-Terrain-Play-v0",
        "LoComposition-Go2-Rough-Terrain-Joint-State-History-v0": "CaT-Go2-Rough-Terrain-Joint-State-History-v0",
        "LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0": "CaT-Go2-Rough-Terrain-Joint-State-History-Play-v0",
        "LoComposition-Go2-Rough-Terrain-Full-State-History-v0": "CaT-Go2-Rough-Terrain-Full-State-History-v0",
        "LoComposition-Go2-Rough-Terrain-Full-State-History-Play-v0": "CaT-Go2-Rough-Terrain-Full-State-History-Play-v0",
        "LoComposition-Anymal-C-Rough-Terrain-v0": "CaT-Anymal-C-Rough-Terrain-v0",
        "LoComposition-Anymal-C-Rough-Terrain-Play-v0": "CaT-Anymal-C-Rough-Terrain-Play-v0",
        "LoComposition-Spot-Rough-Terrain-v0": "CaT-Spot-Rough-Terrain-v0",
        "LoComposition-Spot-Rough-Terrain-Play-v0": "CaT-Spot-Rough-Terrain-Play-v0",
    }
    for new_id, old_id in pairs.items():
        new_spec, old_spec = gym.spec(new_id), gym.spec(old_id)
        assert new_spec.entry_point == old_spec.entry_point
        assert new_spec.kwargs == old_spec.kwargs
finally:
    launcher.app.close()
```

The ten explicit pairs above cover every current project train/play/history
registration. If a new project task is added later, add its pair here in the
same change as the registration.

- [ ] **Step 8: Run naming and registry verification**

Run: `pytest -q tests/test_task_naming.py`

Expected: PASS.

Run: `python scripts/verify_task_aliases.py`

Expected: exit 0 after checking all paired registrations.

- [ ] **Step 9: Commit task and CLI compatibility**

```bash
git add exts/locomposition scripts justfile tests/test_task_naming.py
git commit -m "Add LoComposition tasks with CaT aliases"
```

---

### Task 3: Make environment creation install the renamed project reproducibly

**Files:**
- Create: `tests/test_setup_script.py`
- Modify: `create-isaac-lab-env-uv.sh`
- Modify: `requirements.txt` only if the fresh install proves a missing declared runtime dependency

**Interfaces:**
- Preserves: `./create-isaac-lab-env-uv.sh ENV_NAME`.
- Adds: `--root PATH` and `--repo-source URL_OR_PATH` for isolated verification of the current branch.
- Default repository: `https://github.com/LouKordos/LoComposition.git`.
- Installed extension: `./exts/locomposition`.

- [ ] **Step 1: Write setup-script contract tests**

```python
# tests/test_setup_script.py
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "create-isaac-lab-env-uv.sh"


def test_setup_script_shell_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_setup_script_help_names_overrides():
    result = subprocess.run(
        [str(SCRIPT), "--help"], text=True, capture_output=True, check=True
    )
    assert "--root" in result.stdout
    assert "--repo-source" in result.stdout


def test_setup_script_installs_canonical_extension():
    text = SCRIPT.read_text()
    assert "LouKordos/LoComposition.git" in text
    assert "exts/locomposition" in text
    assert "constraints-as-terminations" not in text
```

- [ ] **Step 2: Run the tests and confirm the help/branding failures**

Run: `pytest -q tests/test_setup_script.py`

Expected: FAIL because `--help` is not implemented and old repository/package paths remain.

- [ ] **Step 3: Add safe argument parsing and branch-source overrides**

Implement this interface before the existing install body:

```bash
usage() {
    echo "Usage: $0 ENV_NAME [--root PATH] [--repo-source URL_OR_PATH]"
}

ENV_ROOT="${LOCOMPOSITION_ENV_ROOT:-$HOME/mamba_env_data}"
REPO_SOURCE="${LOCOMPOSITION_REPO_SOURCE:-https://github.com/LouKordos/LoComposition.git}"
```

Parse `--root` and `--repo-source` without `eval`. Reject an existing non-empty
target directory before invoking `uv init`. Set:

```bash
PROJECT_ROOT="$ENV_ROOT/$ENV_NAME"
USER_REPO_DIR="$PROJECT_ROOT/LoComposition"
```

Clone `REPO_SOURCE` into the explicit target and install
`./exts/locomposition`. Retain pinned versions and the optional Slurm-template
behavior. Remove `uv tool update-shell`; environment creation must not mutate
the user's shell configuration.

- [ ] **Step 4: Run the focused setup tests**

Run: `pytest -q tests/test_setup_script.py`

Expected: PASS.

- [ ] **Step 5: Commit the setup workflow**

```bash
git add create-isaac-lab-env-uv.sh requirements.txt tests/test_setup_script.py
git commit -m "Update environment setup for LoComposition"
```

---

### Task 4: Rename the project-owned ROS 2 stack and add honest launch compatibility

**Files:**
- Create: `tests/test_ros_rebrand_contract.py`
- Move: `sim2real/ros2_ws/src/cat_controller/` to `sim2real/ros2_ws/src/locomposition_controller/`
- Move: `sim2real/ros2_ws/src/cat_bringup/` to `sim2real/ros2_ws/src/locomposition_bringup/`
- Move: `sim2real/ros2_ws/src/cat_state_estimation/` to `sim2real/ros2_ws/src/locomposition_state_estimation/`
- Move: `sim2real/ros2_ws/src/cat_perception_msgs/` to `sim2real/ros2_ws/src/locomposition_perception_msgs/`
- Create: minimal compatibility packages at the three old non-message package paths
- Modify: `sim2real/bootstrap_ros2_ws.sh`
- Modify: `sim2real/build-and-run.sh`
- Modify: `compose.yml`
- Modify: `compose.no-multiarch.yml`
- Modify: `.devcontainer/compose.yml`
- Modify: all moved manifests, CMake targets, include paths, launch files, configs, resource names, descriptions, and C++ namespaces

**Interfaces:**
- Produces: canonical packages `locomposition_controller`, `locomposition_bringup`, `locomposition_state_estimation`, and `locomposition_perception_msgs`.
- Preserves: `ros2 launch cat_bringup bringup.launch.py` and the old controller/state-estimation launch filenames through wrapper packages.
- Breaks deliberately: `cat_perception_msgs/ProcessedElevationMap`; all in-repository producers and consumers move together to `locomposition_perception_msgs/ProcessedElevationMap`.

- [ ] **Step 1: Write the ROS source-tree contract tests**

```python
# tests/test_ros_rebrand_contract.py
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sim2real/ros2_ws/src"


def package_name(directory: str) -> str:
    root = ET.parse(SRC / directory / "package.xml").getroot()
    return root.findtext("name")


def test_canonical_ros_packages_exist():
    expected = {
        "locomposition_controller",
        "locomposition_bringup",
        "locomposition_state_estimation",
        "locomposition_perception_msgs",
    }
    assert {package_name(name) for name in expected} == expected


def test_project_message_type_is_renamed_everywhere():
    canonical = SRC / "locomposition_controller"
    text = "\n".join(
        path.read_text(errors="ignore") for path in canonical.rglob("*") if path.is_file()
    )
    assert "locomposition_perception_msgs" in text
    assert "cat_perception_msgs" not in text


def test_legacy_bringup_is_a_wrapper_not_a_copy():
    wrapper = SRC / "cat_bringup/launch/bringup.launch.py"
    text = wrapper.read_text()
    assert "locomposition_bringup" in text
    assert "IncludeLaunchDescription" in text
```

- [ ] **Step 2: Run the tests and confirm the missing-package failures**

Run: `pytest -q tests/test_ros_rebrand_contract.py`

Expected: FAIL because only the `cat_*` ROS packages exist.

- [ ] **Step 3: Move and consistently rename the four canonical packages**

Use Git-aware directory moves. In the controller, rename:

```text
cat_controller                         -> locomposition_controller
cat_control_node.yaml                 -> locomposition_control_node.yaml
cat_elevation_map_processing_node.yaml -> locomposition_elevation_map_processing_node.yaml
cat_elevation_map_comparison_node.yaml -> locomposition_elevation_map_comparison_node.yaml
cat_control.launch.py                 -> locomposition_control.launch.py
```

Update CMake `project()`, executable targets, install targets, include paths,
`ament_index_cpp::get_package_prefix`, node names, config YAML roots, package
dependencies, C++ message includes/namespaces, and launch `package=` values.
Rename include directory `include/cat_controller` to
`include/locomposition_controller` and update every quoted include.

In the Python packages, update `package_name`, resource marker, `setup.cfg`,
manifest name/dependencies, `FindPackageShare`, and launch descriptions. Change
template maintainer/license descriptions only where accurate; do not invent a
new license.

- [ ] **Step 4: Add launch-only compatibility packages**

Create small ament Python packages for `cat_controller`, `cat_bringup`, and
`cat_state_estimation`. Each old launch file includes the canonical launch with
the same `LaunchDescriptionSource` and forwards all launch arguments:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    source = PathJoinSubstitution(
        [FindPackageShare("locomposition_bringup"), "launch", "bringup.launch.py"]
    )
    return LaunchDescription(
        [IncludeLaunchDescription(PythonLaunchDescriptionSource(source))]
    )
```

For wrappers with declared arguments, forward `context.launch_configurations`
or reproduce the exact `DeclareLaunchArgument` list and pass each
`LaunchConfiguration` explicitly. Verify behavior; do not silently drop custom
arguments.

- [ ] **Step 5: Update Docker/bootstrap entry points**

Change bootstrap copy destinations and includes to
`locomposition_controller`. Change Compose service/container labels to
LoComposition. Make the image configurable as
`${LOCOMPOSITION_SIM2REAL_IMAGE:-loukordos/cat-sim2real:latest}` so the current
published image remains usable until republished. Change displayed launch
commands to `ros2 launch locomposition_bringup bringup.launch.py`.

- [ ] **Step 6: Run source and syntax checks**

Run: `pytest -q tests/test_ros_rebrand_contract.py`

Expected: PASS.

Run: `python -m compileall -q sim2real/ros2_ws/src/*/launch sim2real/ros2_ws/src/*/setup.py`

Expected: exit 0.

Run in the configured ROS 2 environment:

```bash
colcon build --symlink-install --packages-up-to locomposition_bringup cat_bringup
```

Expected: canonical packages and compatibility wrappers build successfully.
Ignore only the explicitly accepted ament lint-test collection issue, not build
or message-generation failures.

- [ ] **Step 7: Commit the ROS migration**

```bash
git add sim2real compose.yml compose.no-multiarch.yml .devcontainer tests/test_ros_rebrand_contract.py
git commit -m "Rename ROS stack to LoComposition"
```

---

### Task 5: Replace the README, split deployment docs, and add paper media

**Files:**
- Create: `tests/test_readme_contract.py`
- Replace: `README.md`
- Create: `CITATION.cff`
- Create: `docs/sim2real.md`
- Create: `docs/migration.md`
- Create: `assets/locomposition-overview.png`
- Create: `assets/cot-and-contact-patterns.png`
- Create: `assets/sim2real-overview.png`
- Create: `assets/terrain-contact-adaptation.png`
- Create: `assets/swing-height-adaptation.png`
- Create: `assets/demos/go2-hardware.gif`
- Create: `assets/demos/anymal-c.gif`
- Create: `assets/demos/spot.gif`
- Remove: `assets/teaser.png` after the new README no longer references the upstream CaT teaser

**Interfaces:**
- README links: project page `https://sites.google.com/view/locomposition`, paper `https://arxiv.org/abs/2606.15896`, and a videos link currently targeting the project page.
- Documentation links: `docs/sim2real.md` and `docs/migration.md`.
- Media placeholders: stable 16:9 GIF paths listed above, each visibly marked as awaiting final media.

- [ ] **Step 1: Write README/documentation contract tests**

```python
# tests/test_readme_contract.py
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_has_project_links_and_attribution():
    text = README.read_text()
    assert "https://arxiv.org/abs/2606.15896" in text
    assert "https://sites.google.com/view/locomposition" in text
    assert "Constraints as Terminations" in text
    assert "Chane-Sane" in text
    assert "docs/sim2real.md" in text


def test_readme_relative_targets_exist():
    text = README.read_text()
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in targets:
        if "://" not in target and not target.startswith("#"):
            assert (ROOT / target.split("#", 1)[0]).exists(), target


def test_readme_does_not_present_repo_as_cat():
    first_heading = README.read_text().splitlines()[0]
    assert first_heading == "# LoComposition"


def test_machine_readable_citation_names_locomposition():
    citation = (ROOT / "CITATION.cff").read_text()
    assert "LoComposition: Terrain-Adaptive Energy-Efficient Quadruped" in citation
    assert "Loukas" in citation
    assert "Kordos" in citation
```

- [ ] **Step 2: Run the tests and confirm the old-README failures**

Run: `pytest -q tests/test_readme_contract.py`

Expected: FAIL because the README title, project links, docs, and assets are old or absent.

- [ ] **Step 3: Export the supplied paper figures for GitHub**

Use lossless PNG exports from the user-provided source:

```bash
pdftoppm -png -singlefile -r 200 /home/kordoslo/Downloads/EfficientLocomotion_ConstraintRL_Loukas/figures/main_figure.pdf assets/locomposition-overview
pdftoppm -png -singlefile -r 200 /home/kordoslo/Downloads/EfficientLocomotion_ConstraintRL_Loukas/figures/cot_patterns_combined.pdf assets/cot-and-contact-patterns
pdftoppm -png -singlefile -r 200 /home/kordoslo/Downloads/EfficientLocomotion_ConstraintRL_Loukas/figures/sim2real_compressed.pdf assets/sim2real-overview
pdftoppm -png -singlefile -r 200 /home/kordoslo/Downloads/EfficientLocomotion_ConstraintRL_Loukas/Rebuttal/Figures/gait_comparison_terrain.pdf assets/terrain-contact-adaptation
pdftoppm -png -singlefile -r 200 /home/kordoslo/Downloads/EfficientLocomotion_ConstraintRL_Loukas/figures/plot_step_height_box_comparison.pdf assets/swing-height-adaptation
```

Inspect each export at original resolution. Crop only surrounding whitespace;
do not alter plot content, colors, labels, or aspect ratios.

- [ ] **Step 4: Create honest media placeholders**

Generate three simple, static 16:9 GIFs with a neutral background and centered
labels `Unitree Go2 hardware — replace with final GIF`, `ANYmal C simulation —
replace with final GIF`, and `Boston Dynamics Spot simulation — replace with
final GIF`. Do not synthesize robot footage. Keep each under 100 KB.

```bash
mkdir -p assets/demos
convert -size 960x540 'xc:#111827' -fill '#E5E7EB' -font DejaVu-Sans -pointsize 32 -gravity center -annotate +0+0 'Unitree Go2 hardware — replace with final GIF' -set delay 100 -loop 0 assets/demos/go2-hardware.gif
convert -size 960x540 'xc:#111827' -fill '#E5E7EB' -font DejaVu-Sans -pointsize 32 -gravity center -annotate +0+0 'ANYmal C simulation — replace with final GIF' -set delay 100 -loop 0 assets/demos/anymal-c.gif
convert -size 960x540 'xc:#111827' -fill '#E5E7EB' -font DejaVu-Sans -pointsize 32 -gravity center -annotate +0+0 'Boston Dynamics Spot simulation — replace with final GIF' -set delay 100 -loop 0 assets/demos/spot.gif
```

Run `identify assets/demos/*.gif` and `du -h assets/demos/*.gif`; each output
must report `960x540`, GIF format, and a size below 100 KB.

- [ ] **Step 5: Write the concise README**

Use this section order and keep each section short:

```markdown
# LoComposition

### Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors

[Project page](https://sites.google.com/view/locomposition) ·
[Paper](https://arxiv.org/abs/2606.15896) ·
[Videos](https://sites.google.com/view/locomposition)

![LoComposition overview](assets/locomposition-overview.png)

## Why LoComposition
## Results at a glance
## Additional evidence
## Installation
## Quick start
## Supported robots
## Repository layout
## Sim-to-real deployment
## Citation and attribution
## License
```

Lead with outcomes, then add technical detail. Include exact tested versions
from the repository. Use one training command with the canonical joint-history
Go2 task, one bounded evaluation command, and one standalone plotting command.
State that the original CaT implementation and Solo12 examples remain for
compatibility and link directly to `tasks/utils/cat`.

Under `License`, state narrowly that source files retain their existing
file-level license notices (predominantly BSD-3-Clause, with Apache-2.0 in the
ROS controller package) and that a repository-wide license has not yet been
declared. Do not present the whole project as MIT, BSD, or Apache until the
owner adds an agreed top-level license.

Keep claims faithful to the paper and user transcript. Use “gait-style,” not
the transcript's speech-to-text error “gate-style.” Distinguish a 76% reduction
from LP to LEP from the paper's 56% reduction relative to RP.

Keep the four non-paper diagnostics under a visibly separate “Additional
evidence” heading so they do not blur into the current arXiv results:

1. Terrain conditioning: swing height changes from 2.28 cm on flat ground to
   5.72 cm on uneven terrain, while simultaneous diagonal contact changes from
   81% to 56%.
2. Delay robustness: with a 20 ms action delay, LoComposition velocity RMSE
   changes from 0.19 to 0.26 m/s, while the no-energy LP ablation changes from
   0.17 to 0.47 m/s.
3. Matched reward-penalty diagnostic: the best tested reward coefficient still
   produces roughly 15 times more torque-limit violations than CaT encoding
   under matched action scale and PPO settings.
4. Embodiment transfer: the same formulation and training recipe produces
   effective separately trained ANYmal C and Spot policies. Action scale and
   actuator/constraint bounds follow each embodiment; mass randomization,
   disturbances, and energy coefficients are deterministically normalized by
   mass ratio rather than selected through an embodiment-specific tuning
   search. Never describe this as one policy transferring across robots.

- [ ] **Step 6: Add machine-readable citation metadata**

Create `CITATION.cff` with CFF 1.2 metadata for the software and a
`preferred-citation` article entry. Use the paper title, all seven supplied
authors in paper order, year 2026, and
`https://arxiv.org/abs/2606.15896`. Keep the README's human-readable BibTeX for
both LoComposition and Chane-Sane et al.'s CaT paper. Do not add a license key:
the repository currently has no single top-level license.

- [ ] **Step 7: Move and edit deployment instructions into `docs/sim2real.md`**

Preserve useful details from the old README under these headings:

```markdown
# Sim-to-real deployment
## Architecture
## Safety boundary
## Prerequisites
## Time synchronization
## Robot and LiDAR networking
## Build the Docker and ROS 2 workspaces
## Configure the MID-360 and elevation mapping
## Launch LoComposition
## Validate topics, transforms, and map quality
## Troubleshooting
```

Use new package names and link back to the README. Retain the warning that the
controller is soft real-time, freshness checks trigger safe stop behavior, and
accurate synchronized time and mapping are required.

- [ ] **Step 8: Write `docs/migration.md`**

Include exact old/new mappings for package imports, task IDs, config module
paths, ROS packages, message types, launch commands, Docker variables, and the
future repository URL. End with this owner checklist:

1. Rename the GitHub repository to `LoComposition`.
2. Update the local `fork` remote URL.
3. Confirm GitHub redirects the old clone URL.
4. Publish a LoComposition Docker image and change the Compose default.
5. Supply the final direct video/blog URLs and three demo GIFs.
6. Choose and add a repository-wide top-level license with the agreement of the
   relevant copyright holders; until then, describe only the existing
   file-level BSD-3-Clause and ROS-package Apache-2.0 licensing.
7. Run the README external-link checker after the rename.

- [ ] **Step 9: Run documentation checks and inspect assets**

Run: `pytest -q tests/test_readme_contract.py`

Expected: PASS.

Run: `file assets/*.png assets/demos/*.gif`

Expected: all files recognized as PNG/GIF, non-empty, with no PDF linked as an inline GitHub image.

- [ ] **Step 10: Commit the public documentation and media**

```bash
git add README.md CITATION.cff docs assets tests/test_readme_contract.py
git commit -m "Present repository as LoComposition"
```

---

### Task 6: Run Isaac-aware compatibility, training, evaluation, and plotting smokes

**Files:**
- Modify only files implicated by a reproduced smoke failure
- Update: `scripts/verify_task_aliases.py` if runtime observations require stricter assertions
- Create: `scripts/smoke_task.py`

**Interfaces:**
- Uses existing environment: `/home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv`.
- Uses real checkpoint run: `/home/kordoslo/dev/constraints-as-terminations/logs/clean_rl/env_new_isaac_lab/2026-08-05-12-32-51`.
- Creates ignored smoke artifacts under `logs/clean_rl/locomposition_rebrand_smoke/` and the existing checkpoint run's evaluation directory.

- [ ] **Step 1: Install this worktree's extension into the existing Isaac environment**

Run from the worktree root:

```bash
uv pip install \
  --python /home/kordoslo/mamba_env_data/env_new_isaac_lab/.venv/bin/python \
  --no-build-isolation \
  --no-deps \
  --editable ./exts/locomposition
```

Expected: exit 0 and the editable distribution resolves to this worktree's
`exts/locomposition` directory.

- [ ] **Step 2: Verify canonical and legacy task registration**

Run in the installed Isaac environment:

```bash
python scripts/verify_task_aliases.py
```

Expected: every canonical/legacy pair resolves identical registry metadata and the app closes cleanly.

- [ ] **Step 3: Construct/reset all canonical embodiment tasks**

Implement `scripts/smoke_task.py` with `--task`, `--steps`, and AppLauncher
arguments. It must launch Isaac before importing Gym task modules, construct
the task with one environment through `parse_env_cfg`, reset it, execute the
requested number of zero-action steps, assert 12 action dimensions and finite
policy observations/rewards, print the resolved task/config/action/observation
summary, close the environment, and close the app in `finally` blocks.

Run each command as a fresh Isaac process:

```bash
python scripts/smoke_task.py --headless --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 --steps=25
python scripts/smoke_task.py --headless --task=LoComposition-Anymal-C-Rough-Terrain-v0 --steps=25
python scripts/smoke_task.py --headless --task=LoComposition-Spot-Rough-Terrain-v0 --steps=25
```

Expected for each: exit 0 after 25 finite steps. Compare the printed resolved
configuration against the corresponding source config so the rename does not
change action scale, mass randomization, disturbance ranges, mass-scaled energy
coefficient, actuator limits, or operational-limit bounds. Do not infer
successful learned locomotion from these startup checks.

- [ ] **Step 4: Run a real two-iteration training smoke**

```bash
ENV_NAME=locomposition_rebrand_smoke \
WANDB_MODE=offline \
OMNICLIENT_HUB_MODE=disabled \
python scripts/clean_rl/train.py \
  --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-v0 \
  --seed=46 \
  --headless \
  --num_envs=64 \
  --num_iterations=2 \
  --logger=tensorboard
```

Expected: exit 0, two finite rollout/update iterations, printed canonical task
name, 12 actions, expected observation shape, and no import/registry/CUDA/NaN
failure.

- [ ] **Step 5: Evaluate a real existing checkpoint through the canonical task**

```bash
python scripts/eval.py \
  --headless \
  --run_dir=/home/kordoslo/dev/constraints-as-terminations/logs/clean_rl/env_new_isaac_lab/2026-08-05-12-32-51 \
  --eval_checkpoint=6299 \
  --task=LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0 \
  --random_sim_step_length=0 \
  --fixed_scenario=stand_still \
  --fixed_command_sim_steps=25 \
  --skip_cot_sweep \
  --num_plot_jobs_in_parallel=1 \
  --plot_job_stagger_delay=0
```

Expected: exit 0, checkpoint loaded, canonical Go2 profile selected, non-empty
`metrics_summary.json`, `plots/sim_data.npz`, video artifact, and automatically
generated plot directory.

- [ ] **Step 6: Verify legacy checkpoint/task compatibility explicitly**

Run the same bounded evaluation with
`--task=CaT-Go2-Rough-Terrain-Joint-State-History-Play-v0` and `--seed=47` so
it writes a distinct evaluation directory.

Expected: exit 0 and equivalent task/configuration resolution.

- [ ] **Step 7: Run plot generation independently**

```bash
python scripts/generate_plots.py \
  --data_file=/home/kordoslo/dev/constraints-as-terminations/logs/clean_rl/env_new_isaac_lab/2026-08-05-12-32-51/eval_checkpoint_6299_seed_46_action_delay_0_scenario_stand_still/plots/sim_data.npz \
  --output_dir=/home/kordoslo/dev/constraints-as-terminations/logs/clean_rl/env_new_isaac_lab/2026-08-05-12-32-51/eval_checkpoint_6299_seed_46_action_delay_0_scenario_stand_still/plots/standalone-smoke \
  --start_step=0 \
  --end_step=25 \
  --job_timeout=300
```

Expected: exit 0 and multiple non-empty PDF/PNG plot artifacts under the
standalone output directory.

- [ ] **Step 8: Diagnose and fix only reproduced failures**

If any smoke fails, use `superpowers:systematic-debugging`: preserve the full
command/output, isolate root cause, add a focused regression test, then make the
smallest source change. Re-run the failing command from the beginning before
continuing.

- [ ] **Step 9: Commit smoke-driven fixes, if any**

If a tracked source or regression-test file changed, inspect
`git diff --name-only`, stage those reported paths explicitly, and commit them
with `git commit -m "Fix LoComposition runtime compatibility"`. Skip this
commit only if no tracked file changed; never stage smoke logs or generated
evaluation artifacts.

---

### Task 7: Prove fresh environment creation and complete the branding audit

**Files:**
- Modify only files implicated by a reproduced fresh-install or audit failure
- Do not commit generated virtual environments, logs, evaluation data, or plots

**Interfaces:**
- Fresh install root: `/tmp/locomposition-env-smoke` after confirming sufficient free space and that the target environment path does not exist.
- Repository source: the completed local worktree path, ensuring the test installs this branch rather than public `main`.

- [ ] **Step 1: Verify sufficient disk space and choose an unused target**

Run read-only checks for available space and confirm the exact target directory
does not exist. Use a name such as
`locomposition_setup_smoke_20260811`; do not remove or reuse an existing
environment.

- [ ] **Step 2: Run the full environment-creation script**

```bash
./create-isaac-lab-env-uv.sh locomposition_setup_smoke_20260811 \
  --root /tmp/locomposition-env-smoke \
  --repo-source /tmp/locomposition-rename-project-and-docs
```

Expected: exit 0 after creating Python 3.11, installing pinned PyTorch and Isaac
Sim, checking out the pinned Isaac Lab commit, installing its packages,
cloning the completed branch, installing `exts/locomposition`, and printing the
LoComposition activation checklist.

- [ ] **Step 3: Verify the newly created environment independently**

```bash
/tmp/locomposition-env-smoke/locomposition_setup_smoke_20260811/.venv/bin/python \
  -c 'import locomposition, cat_envs; print(locomposition.__file__)'
```

Expected: exit 0 and a path under the fresh environment's LoComposition clone.

Run the task-registry verification with the fresh interpreter.

- [ ] **Step 4: Run the focused test suite and syntax checks fresh**

Run:

```bash
pytest -q \
  tests/test_python_rebrand_contract.py \
  tests/test_task_naming.py \
  tests/test_setup_script.py \
  tests/test_ros_rebrand_contract.py \
  tests/test_readme_contract.py
```

Expected: all focused tests pass. Do not run the known irrelevant unrestricted
ament lint collection as a release gate.

Run: `python -m compileall -q exts/locomposition scripts sim2real/ros2_ws/src`

Expected: exit 0.

- [ ] **Step 5: Audit remaining branding references**

Run:

```bash
git grep -n -i -E 'constraints[- _]?as[- _]?terminations|\bCaT\b|cat_envs|CaT-|cat_' -- ':!*.pt' ':!*.png' ':!*.gif'
```

Classify every result into one of these allowed groups:

1. CaT algorithm/module/class terminology.
2. Citation, provenance, or original Solo12 example.
3. Explicit compatibility alias or migration documentation.
4. Legacy published Docker image default awaiting republication.

Rename or rewrite every other result. Do not match or change `torch.cat`.

- [ ] **Step 6: Check the final diff and repository state**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only intentional tracked changes before the final commit; no generated
environment, logs, videos, or plot outputs.

- [ ] **Step 7: Commit any final verification fixes**

If the audit required tracked edits, inspect `git diff --name-only`, stage each
reported source/documentation/test path explicitly, and commit with
`git commit -m "Complete LoComposition compatibility audit"`. Skip this commit
if the worktree is already clean; never stage generated environments, logs,
videos, or plot outputs.

- [ ] **Step 8: Prepare the user handoff**

Report:

- every renamed file/package/interface;
- every compatibility alias retained;
- every remaining intentional CaT reference category;
- exact environment, training, evaluation, plotting, task-registry, ROS, and
  focused-test commands with exit status;
- any unavailable ROS build verification, without calling it successful;
- the final repository-rename checklist;
- the media asset checklist with filename, 16:9 or source aspect ratio, desired
  scene, crop, duration, looping behavior, and direct link target for Go2,
  ANYmal C, Spot, project video, and blog post.

---

## Plan Self-Review Checklist

- Every approved naming boundary is assigned to Tasks 1, 2, or 4.
- README, paper figures, project links, CaT attribution, focused deployment docs,
  and the media handoff are assigned to Task 5.
- Environment creation, training, evaluation, and plot generation each have a
  real command and expected artifact in Tasks 6 and 7.
- The intentional `locomposition_perception_msgs` type migration is tested and
  documented; it is not mislabeled as wire-compatible.
- Python import identity, Gym registration equivalence, checkpoint task-name
  compatibility, and project-wide branding classification each have an
  explicit verification step.
- The known missing-ament pytest collection issue is excluded narrowly; ROS
  build and message-generation failures remain release blockers when the ROS
  environment is available.
