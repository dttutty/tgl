#!/usr/bin/env python3
"""Patch an installed DGL package to avoid GraphBolt eager imports.

This repo's Blackwell stack uses torch 2.10/cu128, but the bootstrap DGL
package we copy in was built against older PyTorch GraphBolt binaries. TGL does
not use DGL's distributed/GraphBolt dataloading stack, so the safe workaround
is to stop importing those modules eagerly from ``import dgl``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class PatchSpec:
    relative_path: str
    original: str
    replacement: str
    marker: str


PATCHES = (
    PatchSpec(
        relative_path="dgl/__init__.py",
        original='if backend_name == "pytorch":\n    from . import distributed\n',
        replacement=(
            "# Blackwell quick-fix: importing dgl.distributed here pulls in\n"
            "# GraphBolt, whose packaged library does not match the torch\n"
            "# 2.10/cu128 runtime used in this repo's bootstrap flow.\n"
            "# TGL does not use dgl.distributed.\n"
            'if False and backend_name == "pytorch":\n'
            "    from . import distributed\n"
        ),
        marker="TGL does not use dgl.distributed.",
    ),
    PatchSpec(
        relative_path="dgl/dataloading/__init__.py",
        original=(
            'if F.get_preferred_backend() == "pytorch":\n'
            "    from .spot_target import *\n"
            "    from .dataloader import *\n"
            "    from .dist_dataloader import *\n"
        ),
        replacement=(
            'if F.get_preferred_backend() == "pytorch":\n'
            "    from .spot_target import *\n"
            "    # Blackwell quick-fix: these eager imports pull in\n"
            "    # dgl.distributed and then GraphBolt. TGL does not rely on\n"
            "    # these loaders.\n"
            "    if False:\n"
            "        from .dataloader import *\n"
            "        from .dist_dataloader import *\n"
        ),
        marker="TGL does not rely on",
    ),
)


def apply_patch(site_packages: Path) -> list[Path]:
    patched_files: list[Path] = []

    for spec in PATCHES:
        path = site_packages / spec.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing DGL file to patch: {path}")

        text = path.read_text()
        if spec.marker in text:
            continue
        if spec.original not in text:
            raise RuntimeError(
                f"Could not find the expected DGL import block in {path}. "
                "The installed DGL layout is different from the one this "
                "quick-fix was written for."
            )

        path.write_text(text.replace(spec.original, spec.replacement, 1))
        patched_files.append(path)

    return patched_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-packages",
        required=True,
        help="Absolute path to the target environment's site-packages directory.",
    )
    args = parser.parse_args()

    site_packages = Path(args.site_packages).resolve()
    if not site_packages.is_dir():
        raise NotADirectoryError(f"site-packages directory not found: {site_packages}")

    patched_files = apply_patch(site_packages)
    if patched_files:
        print("Patched DGL for Blackwell:")
        for path in patched_files:
            print(path)
    else:
        print("DGL Blackwell patch already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
