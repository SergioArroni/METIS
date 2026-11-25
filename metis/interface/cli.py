"""Command-line interface for METIS.

Entry point registered in pyproject.toml as ``metis = "metis.interface.cli:main"``.
"""

import argparse
import sys

from metis.infrastructure.runtime.logging import setup_logging


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

    # --- infer-schema -------------------------------------------------------
    inf_parser = sub.add_parser(
        "infer-schema",
        help="Infer the column types of a CSV with the Phase −1 rules (YAML to review)",
    )
    inf_parser.add_argument("csv", help="Path to the real table (CSV)")
    inf_parser.add_argument("--sep", default=None, help="CSV separator (default: sniffed)")

    # --- version ------------------------------------------------------------
    sub.add_parser("version", help="Show version")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # The library leaves logging to its host; the CLI is the host here.
    setup_logging()

    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "infer-schema":
        return _cmd_infer_schema(args)
    if args.command == "version":
        from metis import __version__

        print(f"METIS {__version__}")
        return 0

    parser.print_help()
    return 0


def _cmd_infer_schema(args: argparse.Namespace) -> int:
    """Print ``data.schema`` in YAML, with each column's notes as comments."""
    import pandas as pd

    from metis.infrastructure.typing import infer_schema

    df = pd.read_csv(args.csv, sep=args.sep, engine="python", encoding="utf-8-sig")
    inferred = infer_schema(df)
    print("schema:")
    for col, entry in inferred.schema.items():
        for note in inferred.notes.get(col, []):
            print(f"  # {col}: {note}")
        fields = ", ".join(f"{k}: {v}" for k, v in entry.items() if v is not None)
        print(f"  {col}: {{{fields}}}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from metis.application.orchestrator import Orchestrator

    try:
        orchestrator = Orchestrator()
        summary = orchestrator.run(args.config)

        composite = summary.aggregates.get("composite_score", "N/A")
        print(f"Evaluation complete — composite score: {composite}")
        for metric, z in (summary.aggregates.get("calibration_warnings") or {}).items():
            print(f"Warning: {metric} is {z:.1f} sd below another real sample")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        from metis.application.pipeline.calibrator import CalibrationStep
        from metis.application.pipeline.loader import DataLoader
        from metis.application.pipeline.preprocessor import DataPreprocessor
        from metis.infrastructure.runtime.config import load_config

        # Same data, defaults, seed and cache as the pipeline's calibration
        # step: the real table read with its separator, IDs stripped and NaN
        # rows dropped, but not cast (the calibration evaluations cast it).
        config = load_config(args.config)
        loader = DataLoader()
        real_raw = loader.load_real(config)
        ref = DataPreprocessor().prepare_reference(real_raw, config, loader.build_spec(config))
        bounds = CalibrationStep().run_calibration(
            ref.real_clean, config, args.config, n_iterations=args.iterations
        )
        print(f"Calibration complete: {bounds.get_summary()}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
