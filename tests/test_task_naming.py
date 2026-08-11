import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_task_naming():
    module_path = ROOT / "scripts" / "task_naming.py"
    assert module_path.is_file(), "task naming helpers have not been implemented"
    spec = importlib.util.spec_from_file_location("task_naming_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_project_task_ids_canonicalize_by_embodiment():
    task_naming = load_task_naming()
    cases = {
        "CaT-Go2-Rough-Terrain-Play-v0": (
            "LoComposition-Go2-Rough-Terrain-Play-v0"
        ),
        "CaT-Anymal-C-Rough-Terrain-v0": (
            "LoComposition-Anymal-C-Rough-Terrain-v0"
        ),
        "CaT-Spot-Rough-Terrain-v0": "LoComposition-Spot-Rough-Terrain-v0",
    }

    for legacy_id, canonical_id in cases.items():
        assert task_naming.canonical_task_id(legacy_id) == canonical_id


def test_canonical_project_task_ids_are_idempotent():
    task_naming = load_task_naming()
    task_id = "LoComposition-Go2-Rough-Terrain-Joint-State-History-v0"

    assert task_naming.canonical_task_id(task_id) == task_id
    assert task_naming.is_locomposition_task(task_id)
    assert task_naming.is_locomposition_go2_task(task_id)


def test_legacy_project_task_ids_are_still_recognized():
    task_naming = load_task_naming()

    assert task_naming.is_locomposition_task("CaT-Spot-Rough-Terrain-Play-v0")
    assert task_naming.is_locomposition_go2_task("CaT-Go2-Rough-Terrain-v0")


def test_original_cat_examples_are_not_rebranded_as_locomposition():
    task_naming = load_task_naming()
    examples = (
        "Isaac-Velocity-CaT-Flat-Solo12-v0",
        "Isaac-Velocity-CaT-Rectangular-Stairs-Solo12-Play-v0",
    )

    for task_id in examples:
        assert task_naming.canonical_task_id(task_id) == task_id
        assert not task_naming.is_locomposition_task(task_id)
