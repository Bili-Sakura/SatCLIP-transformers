#!/usr/bin/env python3
"""Sync src/satclip custom code into self-contained variant subfolders."""

import argparse
import shutil
from pathlib import Path


def sync_variant_code(repo_root: Path, src_pkg: Path) -> None:
    for variant_dir in sorted(repo_root.glob("SatCLIP-*")):
        if not variant_dir.is_dir():
            continue
        dst = variant_dir / "satclip"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src_pkg,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        print(f"Synced {src_pkg} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to BiliSakura/SatCLIP-transformers repository root",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "satclip",
        help="Path to src/satclip package",
    )
    args = parser.parse_args()
    sync_variant_code(args.repo_root, args.src)


if __name__ == "__main__":
    main()
