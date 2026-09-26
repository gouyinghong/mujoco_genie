"""Shared path guards for the isolated PICO-to-G1 data pipeline."""

from __future__ import annotations

from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
TARGET_REPOSITORY_ROOT = PIPELINE_ROOT.parent
PROTECTED_DATASETS_ROOT = (TARGET_REPOSITORY_ROOT / "datasets").resolve()


def refuse_protected_dataset_write(path: Path, *, purpose: str) -> Path:
    """Resolve a write target and reject the target repository's datasets tree."""
    resolved = path.expanduser().resolve()
    if resolved == PROTECTED_DATASETS_ROOT or PROTECTED_DATASETS_ROOT in resolved.parents:
        raise RuntimeError(
            f"Refusing to {purpose} inside protected datasets tree: {resolved}"
        )
    return resolved
