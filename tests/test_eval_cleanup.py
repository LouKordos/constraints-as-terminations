from __future__ import annotations

import sys
import importlib
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

eval_module = importlib.import_module("eval")


class SubscriptionHandle:
    def __init__(self) -> None:
        self.unsubscribed = False

    def unsubscribe(self) -> None:
        self.unsubscribed = True


def test_cleanup_unsubscribes_asset_tracking_before_scene_deletion() -> None:
    assert hasattr(eval_module, "disable_viewport_tracking_before_close")
    handle = SubscriptionHandle()
    controller = SimpleNamespace(
        cfg=SimpleNamespace(origin_type="asset_root"),
        _viewport_camera_update_handle=handle,
    )
    environment = SimpleNamespace(
        unwrapped=SimpleNamespace(viewport_camera_controller=controller)
    )

    eval_module.disable_viewport_tracking_before_close(environment)

    assert controller.cfg.origin_type == "world"
    assert handle.unsubscribed
    assert controller._viewport_camera_update_handle is None


def test_cleanup_accepts_environment_without_viewport_controller() -> None:
    assert hasattr(eval_module, "disable_viewport_tracking_before_close")
    environment = SimpleNamespace(unwrapped=SimpleNamespace())

    eval_module.disable_viewport_tracking_before_close(environment)
