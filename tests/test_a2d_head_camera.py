from pathlib import Path

import mujoco
import numpy as np

from scripts.a2d_head_camera import (
    HEAD_CAMERA_CX,
    HEAD_CAMERA_CY,
    HEAD_CAMERA_FX,
    HEAD_CAMERA_FY,
    HEAD_CAMERA_HEIGHT,
    HEAD_CAMERA_NAME,
    HEAD_CAMERA_POSITION_M,
    HEAD_CAMERA_PRINCIPAL_OFFSET_PX,
    HEAD_CAMERA_QUAT_WXYZ,
    HEAD_CAMERA_WIDTH,
    camera_matrix,
)
from scripts.convert_a2d_to_mjcf import (
    DEFAULT_A2D_URDF,
    convert_a2d_urdf_to_mjcf,
)
from scripts.replay_a2d_head_camera import distortion_maps, hide_closed_head_shell


def test_converter_adds_calibrated_camera_without_hiding_robot_head(
    tmp_path: Path,
) -> None:
    output = tmp_path / "a2d.xml"
    convert_a2d_urdf_to_mjcf(DEFAULT_A2D_URDF, output)
    model = mujoco.MjModel.from_xml_path(str(output))
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME
    )
    head_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "link-pitch_head"
    )
    head_visual_ids = np.flatnonzero(
        (model.geom_bodyid == head_id) & (model.geom_contype == 0)
    )

    assert camera_id >= 0
    assert model.cam_bodyid[camera_id] == head_id
    np.testing.assert_allclose(model.cam_pos[camera_id], HEAD_CAMERA_POSITION_M)
    np.testing.assert_allclose(
        model.cam_quat[camera_id], HEAD_CAMERA_QUAT_WXYZ, atol=1e-8
    )
    np.testing.assert_array_equal(
        model.cam_resolution[camera_id], (HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT)
    )
    np.testing.assert_allclose(
        model.cam_intrinsic[camera_id, 2:] * model.cam_resolution[camera_id]
        / model.cam_sensorsize[camera_id],
        HEAD_CAMERA_PRINCIPAL_OFFSET_PX,
        atol=1e-5,
    )
    assert len(head_visual_ids) == 1
    assert model.geom_group[head_visual_ids[0]] == 1


def test_intrinsic_matrix_and_distortion_maps_match_real_resolution() -> None:
    np.testing.assert_allclose(
        camera_matrix(),
        (
            (HEAD_CAMERA_FX, 0.0, HEAD_CAMERA_CX),
            (0.0, HEAD_CAMERA_FY, HEAD_CAMERA_CY),
            (0.0, 0.0, 1.0),
        ),
    )
    map_x, map_y = distortion_maps()
    assert map_x.shape == (HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH)
    assert map_y.shape == (HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH)


def test_closed_head_shell_is_hidden_only_in_camera_model() -> None:
    model = mujoco.MjModel.from_xml_path(
        "assets/A2D_Omnipicker/A2D_with_box.xml"
    )
    head_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "link-pitch_head"
    )
    visual_id = int(
        np.flatnonzero(
            (model.geom_bodyid == head_id) & (model.geom_contype == 0)
        )[0]
    )
    assert model.geom_group[visual_id] == 1
    hide_closed_head_shell(model)
    assert model.geom_group[visual_id] == 5
