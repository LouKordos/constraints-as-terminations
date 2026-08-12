from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_plots.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("generate_plots_short_rollout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATE_PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATE_PLOTS)


def test_running_average_is_empty_when_rollout_is_shorter_than_window() -> None:
    times = np.arange(25, dtype=float) * 0.02
    values = np.linspace(0.0, 1.0, 25)

    average_times, running_average = GENERATE_PLOTS.compute_running_average(
        times, values, window_size=100
    )

    assert average_times.shape == (0,)
    assert running_average.shape == (0,)


def test_running_average_aligns_to_right_edge_of_complete_windows() -> None:
    times = np.arange(5, dtype=float)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    average_times, running_average = GENERATE_PLOTS.compute_running_average(
        times, values, window_size=3
    )

    np.testing.assert_array_equal(average_times, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(running_average, np.array([2.0, 3.0, 4.0]))
