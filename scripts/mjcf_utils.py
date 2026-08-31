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


def _add_table(worldbody: ET.Element) -> None:
    """Add a 1.4 x 0.9 x 0.8 m white table in front of the robot."""

    table = ET.SubElement(
        worldbody,
        "body",
        {"name": "table", "pos": "0.90 0 0"},
    )
    ET.SubElement(
        table,
        "geom",
        {
            "name": "table_top",
            "type": "box",
            "pos": "0 0 0.77",
            "size": "0.45 0.70 0.03",
            "rgba": "0.92 0.92 0.92 1",
            "group": "2",
            "friction": "1 0.01 0.001",
        },
    )
    for index, (x, y) in enumerate(
        (
            (-0.37, -0.62),
            (-0.37, 0.62),
            (0.37, -0.62),
            (0.37, 0.62),
        )
    ):
        ET.SubElement(
            table,
            "geom",
            {
                "name": f"table_leg_{index}",
                "type": "box",
                "pos": f"{x} {y} 0.37",
                "size": "0.04 0.04 0.37",
                "rgba": "0.92 0.92 0.92 1",
                "group": "2",
                "friction": "1 0.01 0.001",
            },
        )


def _add_dice(
    mjcf_root: ET.Element,
    worldbody: ET.Element,
    asset_root: Path,
    output_directory: Path,
) -> None:
    """Add the textured dice as a free body resting on the table."""

    mesh_path = asset_root / "dice_final.obj"
    texture_path = asset_root / "dice_texture.png"
    for path in (mesh_path, texture_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dice asset does not exist: {path}")

    asset = mjcf_root.find("asset")
    if asset is None:
        raise ValueError("Converted MJCF does not contain an asset section")

    relative_mesh_path = os.path.relpath(mesh_path.resolve(), output_directory.resolve())
    relative_texture_path = os.path.relpath(
        texture_path.resolve(), output_directory.resolve()
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "dice_texture",
            "type": "2d",
            "file": Path(relative_texture_path).as_posix(),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "dice_material",
            "texture": "dice_texture",
            "specular": "0.3",
            "shininess": "0.2",
        },
    )
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "dice_mesh",
            "file": Path(relative_mesh_path).as_posix(),
        },
    )

    dice = ET.SubElement(
        worldbody,
        "body",
        {"name": "dice", "pos": "0.75 0 0.8248"},
    )
    ET.SubElement(dice, "freejoint", {"name": "dice_free_joint"})
    ET.SubElement(
        dice,
        "geom",
        {
            "name": "dice_visual",
            "type": "mesh",
            "mesh": "dice_mesh",
            "material": "dice_material",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        dice,
        "geom",
        {
            "name": "dice_collision",
            "type": "box",
            "size": "0.0248 0.0248 0.0248",
            "density": "100",
            "rgba": "0 0 0 0",
            "group": "3",
            "friction": "1 0.01 0.001",
        },
    )


def _add_cardboard_box_asset(
    mjcf_root: ET.Element,
    worldbody: ET.Element,
    asset_root: Path,
    output_directory: Path,
) -> None:
    """Add the user's textured 24 x 16 x 7 cm cardboard box asset."""

    mesh_path = asset_root / "1.obj"
    texture_path = asset_root / "cardboard_Base_Color.png"
    for path in (mesh_path, texture_path):
        if not path.is_file():
            raise FileNotFoundError(f"Cardboard box asset does not exist: {path}")

    asset = mjcf_root.find("asset")
    if asset is None:
        raise ValueError("Converted MJCF does not contain an asset section")
    relative_mesh_path = os.path.relpath(mesh_path.resolve(), output_directory.resolve())
    relative_texture_path = os.path.relpath(
        texture_path.resolve(), output_directory.resolve()
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "cardboard_box_texture",
            "type": "2d",
            "file": Path(relative_texture_path).as_posix(),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "cardboard_box_material",
            "texture": "cardboard_box_texture",
            "emission": "0.2",
            "specular": "0.05",
            "shininess": "0.02",
        },
    )
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "cardboard_box_mesh",
            "file": Path(relative_mesh_path).as_posix(),
        },
    )

    cardboard_box = ET.SubElement(
        worldbody,
        "body",
        {"name": "cardboard_box", "pos": "1.10 0.40 0.8"},
    )
    ET.SubElement(
        cardboard_box,
        "geom",
        {
            "name": "cardboard_box_visual",
            "type": "mesh",
            "mesh": "cardboard_box_mesh",
            "material": "cardboard_box_material",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "2",
        },
    )
    collision_common = {
        "type": "box",
        "rgba": "0 0 0 0",
        "group": "3",
        "friction": "0.8 0.01 0.001",
    }
    for name, position, size in (
        ("base", (0.0, 0.0, 0.001), (0.12, 0.08, 0.001)),
        ("wall_x_negative", (-0.118, 0.0, 0.035), (0.002, 0.08, 0.035)),
        ("wall_x_positive", (0.118, 0.0, 0.035), (0.002, 0.08, 0.035)),
        ("wall_y_negative", (0.0, -0.078, 0.035), (0.12, 0.002, 0.035)),
        ("wall_y_positive", (0.0, 0.078, 0.035), (0.12, 0.002, 0.035)),
    ):
        ET.SubElement(
            cardboard_box,
            "geom",
            {
                **collision_common,
                "name": f"cardboard_box_collision_{name}",
                "pos": " ".join(_format_float(value) for value in position),
                "size": " ".join(_format_float(value) for value in size),
            },
        )


def _augment_for_replay(
    mjcf_root: ET.Element,
    *,
    include_table: bool = True,
    dice_asset_root: Path | None = None,
    include_cardboard_box: bool = False,
    cardboard_box_asset_root: Path | None = None,
    output_directory: Path | None = None,
) -> None:
    """Add the scene, A2D EEF sites, and target mocap bodies."""

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

    if include_table:
        _add_table(worldbody)
        if dice_asset_root is None or output_directory is None:
            raise ValueError("Dice asset and output paths are required with the table")
        _add_dice(mjcf_root, worldbody, dice_asset_root, output_directory)
        if include_cardboard_box:
            if cardboard_box_asset_root is None:
                raise ValueError("Cardboard box asset path is required")
            _add_cardboard_box_asset(
                mjcf_root,
                worldbody,
                cardboard_box_asset_root,
                output_directory,
            )

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
