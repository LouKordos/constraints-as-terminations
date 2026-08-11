"""Verify canonical LoComposition task IDs and compatibility aliases in Isaac Lab."""

from __future__ import annotations

import importlib

from isaaclab.app import AppLauncher


TASK_PAIRS = {
    "LoComposition-Go2-Rough-Terrain-v0": "CaT-Go2-Rough-Terrain-v0",
    "LoComposition-Go2-Rough-Terrain-Play-v0": (
        "CaT-Go2-Rough-Terrain-Play-v0"
    ),
    "LoComposition-Go2-Rough-Terrain-Joint-State-History-v0": (
        "CaT-Go2-Rough-Terrain-Joint-State-History-v0"
    ),
    "LoComposition-Go2-Rough-Terrain-Joint-State-History-Play-v0": (
        "CaT-Go2-Rough-Terrain-Joint-State-History-Play-v0"
    ),
    "LoComposition-Go2-Rough-Terrain-Full-State-History-v0": (
        "CaT-Go2-Rough-Terrain-Full-State-History-v0"
    ),
    "LoComposition-Go2-Rough-Terrain-Full-State-History-Play-v0": (
        "CaT-Go2-Rough-Terrain-Full-State-History-Play-v0"
    ),
    "LoComposition-Anymal-C-Rough-Terrain-v0": (
        "CaT-Anymal-C-Rough-Terrain-v0"
    ),
    "LoComposition-Anymal-C-Rough-Terrain-Play-v0": (
        "CaT-Anymal-C-Rough-Terrain-Play-v0"
    ),
    "LoComposition-Spot-Rough-Terrain-v0": "CaT-Spot-Rough-Terrain-v0",
    "LoComposition-Spot-Rough-Terrain-Play-v0": (
        "CaT-Spot-Rough-Terrain-Play-v0"
    ),
}

CONFIG_MODULE_PAIRS = {
    "go2": (
        "locomposition.tasks.locomotion.velocity.config.solo12."
        "locomposition_go2_rough_terrain_env_cfg",
        "locomposition.tasks.locomotion.velocity.config.solo12."
        "cat_go2_rough_terrain_env_cfg",
        "Go2RoughTerrainEnvCfg",
    ),
    "anymal_c": (
        "locomposition.tasks.locomotion.velocity.config.solo12."
        "locomposition_anymal_c_rough_terrain_env_cfg",
        "locomposition.tasks.locomotion.velocity.config.solo12."
        "cat_anymal_c_rough_terrain_env_cfg",
        "AnymalCRoughTerrainEnvCfg",
    ),
    "spot": (
        "locomposition.tasks.locomotion.velocity.config.solo12."
        "locomposition_spot_rough_terrain_env_cfg",
        "locomposition.tasks.locomotion.velocity.config.solo12."
        "cat_spot_rough_terrain_env_cfg",
        "SpotRoughTerrainEnvCfg",
    ),
}


def main() -> None:
    launcher = AppLauncher(headless=True)
    try:
        import gymnasium as gym
        import locomposition.tasks.locomotion.velocity.config.solo12  # noqa: F401

        for canonical_id, legacy_id in TASK_PAIRS.items():
            canonical_spec = gym.spec(canonical_id)
            legacy_spec = gym.spec(legacy_id)
            assert canonical_spec.entry_point == legacy_spec.entry_point
            assert canonical_spec.kwargs == legacy_spec.kwargs
            print(
                f"verified task alias: {legacy_id} -> {canonical_id}", flush=True
            )

        for embodiment, (canonical_name, legacy_name, class_name) in (
            CONFIG_MODULE_PAIRS.items()
        ):
            canonical_module = importlib.import_module(canonical_name)
            legacy_module = importlib.import_module(legacy_name)
            assert getattr(canonical_module, class_name) is getattr(
                legacy_module, class_name
            )
            print(f"verified config alias: {embodiment}", flush=True)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
