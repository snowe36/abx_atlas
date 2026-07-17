"""Console entry points: abx-download, abx-atlas, abx-qsar."""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ChEMBL antibacterial activities")
    parser.add_argument(
        "--max-per-organism",
        type=int,
        default=5000,
        help="Cap activities per organism (default 5000; use 0 for uncapped)",
    )
    parser.add_argument("--no-np", action="store_true", help="Skip natural-product flag fetch")
    parser.add_argument(
        "--all-organisms",
        action="store_true",
        help="Use full Gram+/− organism list (slower) instead of priority ESKAPE set",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from abxatlas.data.curate import curate_activities
    from abxatlas.data.download import download_antibacterial

    cap = None if args.max_per_organism == 0 else args.max_per_organism
    download_antibacterial(
        max_per_organism=cap,
        fetch_np=not args.no_np,
        priority_only=not args.all_organisms,
    )
    curate_activities()
    return 0


def atlas_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build chemical-space atlas figures")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from abxatlas.atlas.run import run_atlas

    run_atlas()
    return 0


def qsar_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run scaffold-split Gram-negative QSAR")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    from abxatlas.models.run import run_qsar

    run_qsar(test_size=args.test_size)
    return 0


if __name__ == "__main__":
    # Allow python -m abxatlas.cli download|atlas|qsar
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd == "download":
        raise SystemExit(download_main(rest))
    if cmd == "atlas":
        raise SystemExit(atlas_main(rest))
    if cmd == "qsar":
        raise SystemExit(qsar_main(rest))
    print("Usage: python -m abxatlas.cli [download|atlas|qsar]", file=sys.stderr)
    raise SystemExit(2)
