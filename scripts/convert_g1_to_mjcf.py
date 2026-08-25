#!/usr/bin/env python3
"""Convert the convex-decomposition G1 URDF into a replay-ready MJCF.

The source URDF contains one non-finite inertial origin and four positive
semidefinite inertia matrices. MuJoCo rejects those values, so this converter
repairs them in memory, leaving the source asset untouched.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    REPO_ROOT
    / "assets"
    / "robot_g1_arms_convex_decomposition"
    / "robot_g1_arms_convex_decomposition.urdf"
)
DEFAULT_MJCF = DEFAULT_URDF.with_suffix(".xml")

ARM_JOINT_NAMES = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)

G1_GRIPPER_MIMIC_CONSTRAINTS = (
    (
        "left_gripper_mimic",
        "idx31_gripper_l_inner_joint1",
        "idx41_gripper_l_outer_joint1",
    ),
    (
        "right_gripper_mimic",
        "idx71_gripper_r_inner_joint1",
        "idx81_gripper_r_outer_joint1",
    ),
)


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    repaired_inertials: tuple[str, ...]
    nq: int
    nv: int
    njnt: int
    ngeom: int
    visual_geoms: int
    collision_geoms: int


def _parse_vector(value: str, expected_size: int) -> np.ndarray:
    vector = np.fromstring(value, sep=" ", dtype=float)
    if vector.shape != (expected_size,):
        raise ValueError(f"Expected {expected_size} values, got {value!r}")
    return vector


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _install_mujoco_compiler_options(robot: ET.Element) -> None:
    """Install options before MjSpec parses the URDF.

    Setting discardvisual on an already parsed MjSpec is too late: the URDF
    importer has already discarded the visual geometry at that point.
    """

    extension = robot.find("mujoco")
    if extension is None:
        extension = ET.Element("mujoco")
        robot.insert(0, extension)

    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.set("fusestatic", "false")
    compiler.set("discardvisual", "false")
    compiler.set("balanceinertia", "true")


def _repair_inertials(robot: ET.Element, min_eigenvalue: float) -> tuple[str, ...]:
    repaired: list[str] = []
    inertia_keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")

    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue

        link_name = link.get("name", "<unnamed>")
        link_was_repaired = False
        origin = inertial.find("origin")
        if origin is not None:
            xyz = _parse_vector(origin.get("xyz", "0 0 0"), 3)
            if not np.isfinite(xyz).all():
                origin.set("xyz", "0 0 0")
                link_was_repaired = True

        inertia = inertial.find("inertia")
        if inertia is not None:
            values = {key: float(inertia.get(key, "nan")) for key in inertia_keys}
            matrix = np.array(
                [
                    [values["ixx"], values["ixy"], values["ixz"]],
                    [values["ixy"], values["iyy"], values["iyz"]],
                    [values["ixz"], values["iyz"], values["izz"]],
                ],
                dtype=float,
            )
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite inertia matrix on link {link_name}")

            eigenvalues = np.linalg.eigvalsh(matrix)
            if eigenvalues[0] < min_eigenvalue:
                matrix += np.eye(3) * (min_eigenvalue - eigenvalues[0])
                for key, value in (
                    ("ixx", matrix[0, 0]),
                    ("ixy", matrix[0, 1]),
                    ("ixz", matrix[0, 2]),
                    ("iyy", matrix[1, 1]),
                    ("iyz", matrix[1, 2]),
                    ("izz", matrix[2, 2]),
                ):
                    inertia.set(key, _format_float(float(value)))
                link_was_repaired = True

        if link_was_repaired:
            repaired.append(link_name)

    return tuple(repaired)


def _load_assets(asset_root: Path) -> dict[str, bytes]:
    mesh_root = asset_root / "meshes"
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"Mesh directory does not exist: {mesh_root}")
    return {
        path.relative_to(asset_root).as_posix(): path.read_bytes()
        for path in sorted(mesh_root.rglob("*"))
        if path.is_file()
    }


def _body(mjcf_root: ET.Element, name: str) -> ET.Element:
    body = mjcf_root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"Converted MJCF is missing body {name!r}")
    return body


def _rewrite_mesh_paths(
    mjcf_root: ET.Element, source_asset_root: Path, output_directory: Path
) -> None:
    """Make mesh paths valid even when --output is outside the asset folder."""

    for mesh in mjcf_root.findall("./asset/mesh"):
        file_name = mesh.get("file")
        if not file_name:
            continue
        source_path = (source_asset_root / file_name).resolve()
        relative_path = os.path.relpath(source_path, output_directory.resolve())
        mesh.set("file", Path(relative_path).as_posix())


def _augment_for_replay(
    mjcf_root: ET.Element,
    *,
    eef_body_names: tuple[str, str] = ("arm_l_end_link", "arm_r_end_link"),
    gripper_mimic_constraints: tuple[tuple[str, str, str], ...] = (
        G1_GRIPPER_MIMIC_CONSTRAINTS
    ),
) -> None:
    option = mjcf_root.find("option")
    if option is None:
        option = ET.Element("option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
        compiler = mjcf_root.find("compiler")
        insert_at = list(mjcf_root).index(compiler) + 1 if compiler is not None else 0
        mjcf_root.insert(insert_at, option)
    else:
        option.set("timestep", "0.002")

    statistic = mjcf_root.find("statistic")
    if statistic is None:
        statistic = ET.Element("statistic", {"center": "0 0 0.8", "extent": "1.4"})
        worldbody = mjcf_root.find("worldbody")
        insert_at = list(mjcf_root).index(worldbody) if worldbody is not None else 0
        mjcf_root.insert(insert_at, statistic)

    visual = mjcf_root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        worldbody = mjcf_root.find("worldbody")
        insert_at = list(mjcf_root).index(worldbody) if worldbody is not None else 0
        mjcf_root.insert(insert_at, visual)
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {"ambient": "0.35 0.35 0.35", "diffuse": "0.75 0.75 0.75"},
        )
    if visual.find("rgba") is None:
        ET.SubElement(visual, "rgba", {"contactpoint": "1 0.25 0.1 1"})

    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        raise ValueError("Converted MJCF does not contain a worldbody")

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "replay_floor",
            "type": "plane",
            "pos": "0 0 -0.01",
            "size": "3 3 0.1",
            "rgba": "0.18 0.2 0.23 1",
            "group": "2",
            "friction": "1 0.01 0.001",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": "1.5 -1.5 3",
            "dir": "-0.3 0.3 -1",
            "diffuse": "0.8 0.8 0.8",
        },
    )

    for body_name, site_name, rgba in (
        (eef_body_names[0], "left_eef_actual", "0.1 0.9 0.2 1"),
        (eef_body_names[1], "right_eef_actual", "0.1 0.9 0.2 1"),
    ):
        ET.SubElement(
            _body(mjcf_root, body_name),
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "size": "0.012",
                "rgba": rgba,
                "group": "2",
            },
        )

    for side in ("left", "right"):
        target_body = ET.SubElement(
            worldbody,
            "body",
            {"name": f"{side}_eef_target_body", "mocap": "true", "pos": "0 0 -10"},
        )
        ET.SubElement(
            target_body,
            "site",
            {
                "name": f"{side}_eef_target",
                "type": "sphere",
                "size": "0.009",
                "rgba": "0.95 0.15 0.1 0.9",
                "group": "2",
            },
        )

    equality = mjcf_root.find("equality")
    if gripper_mimic_constraints and equality is None:
        equality = ET.SubElement(mjcf_root, "equality")
    for constraint_name, inner, outer in gripper_mimic_constraints:
        assert equality is not None
        ET.SubElement(
            equality,
            "joint",
            {
                "name": constraint_name,
                "joint1": inner,
                "joint2": outer,
                "polycoef": "0 -1 0 0 0",
            },
        )


def _validate_model(model: mujoco.MjModel) -> tuple[int, int]:
    missing = [
        name
        for name in ARM_JOINT_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
    ]
    if missing:
        raise ValueError(f"Converted model is missing arm joints: {missing}")

    visual_geoms = int(
        np.count_nonzero((model.geom_group == 1) & (model.geom_contype == 0))
    )
    collision_geoms = int(
        np.count_nonzero((model.geom_group == 0) & (model.geom_contype != 0))
    )
    if visual_geoms != 61:
        raise ValueError(f"Expected 61 visual geoms, found {visual_geoms}")
    if collision_geoms != 32:
        raise ValueError(f"Expected 32 robot collision geoms, found {collision_geoms}")
    return visual_geoms, collision_geoms


def convert_urdf_to_mjcf(
    urdf_path: Path = DEFAULT_URDF,
    output_path: Path = DEFAULT_MJCF,
    *,
    min_inertia_eigenvalue: float = 1e-10,
) -> ConversionResult:
    urdf_path = urdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    if min_inertia_eigenvalue <= 0:
        raise ValueError("min_inertia_eigenvalue must be positive")

    robot = ET.parse(urdf_path).getroot()
    if robot.tag != "robot":
        raise ValueError(f"Expected a URDF <robot> root, found <{robot.tag}>")

    _install_mujoco_compiler_options(robot)
    repaired = _repair_inertials(robot, min_inertia_eigenvalue)
    assets = _load_assets(urdf_path.parent)

    spec = mujoco.MjSpec.from_string(
        ET.tostring(robot, encoding="unicode"), assets=assets
    )
    spec.modelname = "g1_arms_convex_replay"
    # Compile once before serialization to fail early on invalid URDF data.
    spec.compile()

    mjcf_root = ET.fromstring(spec.to_xml())
    _rewrite_mesh_paths(mjcf_root, urdf_path.parent, output_path.parent)
    _augment_for_replay(mjcf_root)
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
        visual_geoms, collision_geoms = _validate_model(model)
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
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--min-inertia-eigenvalue", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = convert_urdf_to_mjcf(
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
