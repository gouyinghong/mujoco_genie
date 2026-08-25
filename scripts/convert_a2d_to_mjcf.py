#!/usr/bin/env python3
"""Convert assets/A2D_Omnipicker/A2D.urdf into a replay-ready MJCF."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_g1_to_mjcf import (
    ConversionResult,
    _augment_for_replay,
    _install_mujoco_compiler_options,
    _load_assets,
    _repair_inertials,
    _rewrite_mesh_paths,
)


DEFAULT_A2D_URDF = REPO_ROOT / "assets" / "A2D_Omnipicker" / "A2D.urdf"
DEFAULT_A2D_MJCF = DEFAULT_A2D_URDF.with_suffix(".xml")

A2D_ARM_JOINT_NAMES = tuple(
    [f"Joint{index}_l" for index in range(1, 8)]
    + [f"Joint{index}_r" for index in range(1, 8)]
)


def _validate_a2d_model(model: mujoco.MjModel) -> tuple[int, int]:
    missing = [
        name
        for name in A2D_ARM_JOINT_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
    ]
    if missing:
        raise ValueError(f"Converted A2D model is missing arm joints: {missing}")

    visual_geoms = int(
        np.count_nonzero((model.geom_group == 1) & (model.geom_contype == 0))
    )
    collision_geoms = int(
        np.count_nonzero((model.geom_group == 0) & (model.geom_contype != 0))
    )
    if visual_geoms != 39:
        raise ValueError(f"Expected 39 A2D visual geoms, found {visual_geoms}")
    if collision_geoms != 39:
        raise ValueError(f"Expected 39 A2D collision geoms, found {collision_geoms}")
    return visual_geoms, collision_geoms


def convert_a2d_urdf_to_mjcf(
    urdf_path: Path = DEFAULT_A2D_URDF,
    output_path: Path = DEFAULT_A2D_MJCF,
    *,
    min_inertia_eigenvalue: float = 1e-10,
) -> ConversionResult:
    urdf_path = urdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"A2D URDF does not exist: {urdf_path}")

    robot = ET.parse(urdf_path).getroot()
    if robot.tag != "robot":
        raise ValueError(f"Expected a URDF <robot> root, found <{robot.tag}>")
    _install_mujoco_compiler_options(robot)
    repaired = _repair_inertials(robot, min_inertia_eigenvalue)
    assets = _load_assets(urdf_path.parent)

    spec = mujoco.MjSpec.from_string(
        ET.tostring(robot, encoding="unicode"),
        assets=assets,
    )
    spec.modelname = "a2d_omnipicker_replay"
    spec.compile()

    mjcf_root = ET.fromstring(spec.to_xml())
    _rewrite_mesh_paths(mjcf_root, urdf_path.parent, output_path.parent)
    _augment_for_replay(
        mjcf_root,
        eef_body_names=("Link7_l", "Link7_r"),
        gripper_mimic_constraints=(),
    )
    ET.indent(mjcf_root, space="  ")
    xml_bytes = ET.tostring(mjcf_root, encoding="utf-8", xml_declaration=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", dir=output_path.parent, delete=False
        ) as temporary:
            temporary.write(xml_bytes)
            temporary_path = Path(temporary.name)
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
        visual_geoms, collision_geoms = _validate_a2d_model(model)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ConversionResult(
        output_path=output_path,
        repaired_inertials=repaired,
        nq=model.nq,
        nv=model.nv,
        njnt=model.njnt,
        ngeom=model.ngeom,
        visual_geoms=visual_geoms,
        collision_geoms=collision_geoms,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_A2D_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_A2D_MJCF)
    parser.add_argument("--min-inertia-eigenvalue", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = convert_a2d_urdf_to_mjcf(
        args.urdf,
        args.output,
        min_inertia_eigenvalue=args.min_inertia_eigenvalue,
    )
    repaired = ", ".join(result.repaired_inertials) or "none"
    print(f"Generated: {result.output_path}")
    print(f"Repaired inertials: {repaired}")
    print(
        "Model: "
        f"nq={result.nq}, nv={result.nv}, njnt={result.njnt}, "
        f"ngeom={result.ngeom}, visual={result.visual_geoms}, "
        f"robot_collision={result.collision_geoms}"
    )


if __name__ == "__main__":
    main()
