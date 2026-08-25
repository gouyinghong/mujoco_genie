"""Shared helpers for converting the A2D URDF to MJCF."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


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
    """Preserve visual geometry while importing URDF through MjSpec."""

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
    """Make mesh paths valid when output is outside the asset directory."""

    for mesh in mjcf_root.findall("./asset/mesh"):
        file_name = mesh.get("file")
        if not file_name:
            continue
        source_path = (source_asset_root / file_name).resolve()
        relative_path = os.path.relpath(source_path, output_directory.resolve())
        mesh.set("file", Path(relative_path).as_posix())


def _augment_for_replay(mjcf_root: ET.Element) -> None:
    """Add the floor, lighting, A2D EEF sites and target mocap bodies."""

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

    for body_name, site_name in (
        ("Link7_l", "left_eef_actual"),
        ("Link7_r", "right_eef_actual"),
    ):
        ET.SubElement(
            _body(mjcf_root, body_name),
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "size": "0.012",
                "rgba": "0.1 0.9 0.2 1",
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
