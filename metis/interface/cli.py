"""Command-line interface for METIS.

Entry point registered in pyproject.toml as ``metis = "metis.interface.cli:main"``.
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metis",
        description="METIS — Modular framework for synthetic tabular data evaluation.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- evaluate -----------------------------------------------------------
    eval_parser = sub.add_parser("evaluate", help="Run evaluation pipeline")
    eval_parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")

    # --- calibrate ----------------------------------------------------------
    cal_parser = sub.add_parser("calibrate", help="Run calibration")
    cal_parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    cal_parser.add_argument(
        "--iterations", "-n", type=int, default=None, help="Override n_iterations"
    )

    # --- version ------------------------------------------------------------
    sub.add_parser("version", help="Show version")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "version":
        from metis import __version__

        print(f"METIS {__version__}")
        return 0

    parser.print_help()
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from metis.application.orchestrator import Orchestrator

    try:
        orchestrator = Orchestrator()
        summary = orchestrator.run(args.config)

        composite = summary.aggregates.get("composite_score", "N/A")
        print(f"Evaluation complete — composite score: {composite}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        from metis.calibrate.cache.cache_manager import CacheManager
        from metis.infrastructure.io.loaders import load_csv
        from metis.infrastructure.runtime.config import load_config

        config = load_config(args.config)
        data_cfg = config["data"]
        real_data = load_csv(data_cfg["real"])

        cal_cfg = config.get("calibration", {})
        n_iter = args.iterations or cal_cfg.get("n_iterations", 10)

        mgr = CacheManager()
        bounds = mgr.get_or_calibrate(
            real_data=real_data,
            config_path=args.config,
            n_iterations=n_iter,
            sample_percentage=cal_cfg.get("sample_percentage", 10.0),
            base_seed=cal_cfg.get("base_seed", 42),
            n_jobs=cal_cfg.get("n_jobs", 1),
        )
        print(f"Calibration complete: {bounds.get_summary()}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
