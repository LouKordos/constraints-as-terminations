from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_plots


def _assert_complete_joint_layout(joint_names, expected_rows):
    leg_rows, leg_cols, foot_from_joint = generate_plots._build_joint_layout(joint_names)

    assert leg_rows == expected_rows
    assert leg_cols == [0] * 4 + [1] * 4 + [2] * 4
    assert foot_from_joint == expected_rows
    assert len(set(zip(leg_rows, leg_cols))) == len(joint_names)


def test_build_joint_layout_preserves_go2_mapping():
    joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]

    _assert_complete_joint_layout(joint_names, [0, 1, 2, 3] * 3)


def test_build_joint_layout_supports_anymal_mapping():
    joint_names = [
        "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
        "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
        "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
    ]

    _assert_complete_joint_layout(joint_names, [0, 2, 1, 3] * 3)
