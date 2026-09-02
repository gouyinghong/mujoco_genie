"""Calibration constants and MJCF support for the real A2D head camera."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np


HEAD_CAMERA_NAME = "head_color"
HEAD_CAMERA_WIDTH = 1280
HEAD_CAMERA_HEIGHT = 800
HEAD_CAMERA_FPS = 30.0

# /data/parameters/head_intrinsic_params.json on the real robot.
HEAD_CAMERA_FX = 650.3485107421875
HEAD_CAMERA_FY = 649.6729125976562
HEAD_CAMERA_CX = 643.7561645507812
HEAD_CAMERA_CY = 401.63385009765625
# MJCF's principalpixel is an offset from the image center, whereas OpenCV's
# cx/cy values are absolute pixel coordinates measured from the top-left.
HEAD_CAMERA_PRINCIPAL_OFFSET_PX = (
    HEAD_CAMERA_CX - HEAD_CAMERA_WIDTH / 2.0,
    HEAD_CAMERA_CY - HEAD_CAMERA_HEIGHT / 2.0,
)
HEAD_CAMERA_DISTORTION = np.array(
    (
        -0.04981835186481476,
        0.05437535047531128,
        -0.0011401977390050888,
        -0.0003914425615221262,
        -0.016374263912439346,
    ),
    dtype=float,
)

# /data/parameters/head_extrinsic_params.json.  The rotation stored there is
# camera-optical -> head_pitch.  OpenCV looks along optical +Z with +Y down;
# MuJoCo looks along camera -Z with +Y up, hence the 180-degree X conversion
# already included in this MuJoCo quaternion.
HEAD_CAMERA_POSITION_M = (
    -0.08377510539410596,
    0.05589438324345217,
    -0.011301675696943412,
)
HEAD_CAMERA_QUAT_WXYZ = (
    0.70644127,
    0.00596275,
    0.70771816,
    -0.00633991,
)


def camera_matrix() -> np.ndarray:
    return np.array(
        (
            (HEAD_CAMERA_FX, 0.0, HEAD_CAMERA_CX),
            (0.0, HEAD_CAMERA_FY, HEAD_CAMERA_CY),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def add_calibrated_head_camera(mjcf_root: ET.Element) -> None:
    """Attach the calibrated camera without changing robot visual groups."""

    head = mjcf_root.find(".//body[@name='link-pitch_head']")
    if head is None:
        raise ValueError("Converted A2D model is missing body 'link-pitch_head'")
    existing = head.find(f"./camera[@name='{HEAD_CAMERA_NAME}']")
    if existing is not None:
        head.remove(existing)
    ET.SubElement(
        head,
        "camera",
        {
            "name": HEAD_CAMERA_NAME,
            "mode": "fixed",
            "pos": " ".join(f"{value:.12g}" for value in HEAD_CAMERA_POSITION_M),
            "quat": " ".join(f"{value:.12g}" for value in HEAD_CAMERA_QUAT_WXYZ),
            "focalpixel": f"{HEAD_CAMERA_FX:.12g} {HEAD_CAMERA_FY:.12g}",
            "principalpixel": " ".join(
                f"{value:.12g}" for value in HEAD_CAMERA_PRINCIPAL_OFFSET_PX
            ),
            "resolution": f"{HEAD_CAMERA_WIDTH} {HEAD_CAMERA_HEIGHT}",
            "sensorsize": "1 1",
        },
    )
