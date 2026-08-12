# Migration to LoComposition names

This branch changes the public project identity while keeping older Python imports and Gym task IDs usable. The CaT name remains wherever it refers specifically to the Constraints-as-Terminations algorithm.

## Name mapping

| Previous name | Canonical name | Compatibility |
| --- | --- | --- |
| Repository `constraints-as-terminations` | `LoComposition` | GitHub normally redirects the old URL after the owner renames the repository; verify this after the rename |
| Python distribution/package `cat_envs` | `locomposition` | `import cat_envs` and matching submodule imports are retained as aliases to the canonical module objects |
| `CaT-Go2-*` | `LoComposition-Go2-*` | Old Gym IDs remain registered |
| `CaT-Anymal-C-*` | `LoComposition-Anymal-C-*` | Old Gym IDs remain registered |
| `CaT-Spot-*` | `LoComposition-Spot-*` | Old Gym IDs remain registered |
| `cat_*_rough_terrain_env_cfg.py` | `locomposition_*_rough_terrain_env_cfg.py` | Old modules forward to the canonical modules |
| `cat_controller` | `locomposition_controller` | The old ROS package is a launch-only wrapper; C++ consumers must use the canonical package |
| `cat_bringup` | `locomposition_bringup` | Old launch command forwards to the canonical launch file |
| `cat_state_estimation` | `locomposition_state_estimation` | Old launch command forwards to the canonical launch files |
| `cat_perception_msgs/ProcessedElevationMap` | `locomposition_perception_msgs/ProcessedElevationMap` | **Breaking:** ROS message package identity and generated type support changed; rebuild every publisher/subscriber in the same workspace |

The Solo12 tasks named `Isaac-Velocity-CaT-*` are preserved as CaT examples. They describe the constraint mechanism itself and are not LoComposition experiments.

## Updating Python and task callers

New code should import the canonical package:

```python
from locomposition.tasks.utils.cat import ConstraintManager
```

Old imports continue to resolve to the same module objects during the compatibility window:

```python
from cat_envs.tasks.utils.cat import ConstraintManager
```

Use `LoComposition-Go2-Rough-Terrain-Joint-State-History-v0` for main-policy training and the matching `-Play-v0` task for evaluation. Existing run scripts using the corresponding `CaT-*` IDs do not need an immediate rewrite.

## Updating ROS workspaces

Delete the workspace's generated `build/`, `install/`, and `log/` directories before rebuilding after the message rename. Source code that mentions the old message package must change both its dependency and include/namespace:

```text
cat_perception_msgs/msg/processed_elevation_map.hpp
cat_perception_msgs::msg::ProcessedElevationMap
```

becomes:

```text
locomposition_perception_msgs/msg/processed_elevation_map.hpp
locomposition_perception_msgs::msg::ProcessedElevationMap
```

The canonical stack starts with:

```bash
ros2 launch locomposition_bringup bringup.launch.py
```

The old `ros2 launch cat_bringup bringup.launch.py` entry point remains as a thin forwarding wrapper. It does not restore the old ROS message type identity.

## Container image

The Compose service is named `locomposition_sim2real` and its default image is `loukordos/locomposition-sim2real:latest`. Override the registry, owner, or tag without editing Compose files:

```bash
export LOCOMPOSITION_SIM2REAL_IMAGE=owner/locomposition-sim2real:tag
```

## Repository-owner checklist

1. Rename the GitHub repository to `LoComposition` only after this branch is ready to become the default branch.
2. Update the fork/upstream remotes and check the clone URL shown by GitHub.
3. Verify that the old GitHub URL redirects and that badges, archived scripts, and external project-page links still resolve.
4. Build or publish `loukordos/locomposition-sim2real:latest`, or set `LOCOMPOSITION_SIM2REAL_IMAGE` to the image name you prefer.
5. Replace the three labelled GIF placeholders and add the final personal-blog URL listed in the [asset inventory](assets.md). The direct project-video URL is already present.
6. Confirm the intended licensing for the LoComposition contributions. Decide whether the inherited `LICENCE` applies; otherwise add an agreed top-level license without replacing existing file-level notices.
7. Re-run the repository tests and an external-link check after the GitHub rename is live.
