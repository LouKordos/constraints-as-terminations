"""Run the installed Isaac Lab RSL-RL trainer with local CaT tasks registered.

The installed trainer intentionally contains an extension-template placeholder.
This bootstrap replaces only that marker in memory, after AppLauncher startup,
so the repository task package is imported at the point intended for external
extensions. The upstream file on disk remains untouched.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import isaaclab


_EXTENSION_MARKER = "# PLACEHOLDER: Extension template (do not remove this comment)"
_EXTENSION_IMPORT = "import locomposition.tasks  # noqa: F401"


def _installed_trainer_path() -> Path:
    isaaclab_root = Path(inspect.getfile(isaaclab)).resolve().parents[3]
    trainer_path = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(f"Installed Isaac Lab RSL-RL trainer not found: {trainer_path}")
    return trainer_path


def main() -> None:
    trainer_path = _installed_trainer_path()
    source = trainer_path.read_text()

    marker_count = source.count(_EXTENSION_MARKER)
    if marker_count != 1:
        raise RuntimeError(
            "Expected exactly one Isaac Lab extension placeholder in "
            f"{trainer_path}, found {marker_count}"
        )

    source = source.replace(
        _EXTENSION_MARKER,
        f"{_EXTENSION_IMPORT}\n\n{_EXTENSION_MARKER}",
        1,
    )

    # Preserve upstream cli_args resolution and diagnostics as if train.py had
    # been launched directly.
    sys.path.insert(0, str(trainer_path.parent))
    sys.argv[0] = str(trainer_path)
    namespace = {
        "__file__": str(trainer_path),
        "__name__": "__main__",
        "__package__": None,
    }
    exec(compile(source, str(trainer_path), "exec"), namespace)


if __name__ == "__main__":
    main()
