"""Command-line entry point for nested-fold batch experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from batch_runner import DEFAULT_ALGORITHMS, BatchConfig, run_batch


def _csv_values(value: str) -> tuple[str, ...]:
    """Parse a comma-separated non-empty tuple."""
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated value must not be empty")
    return values


def main() -> None:
    """Parse batch options and write one JSON result list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="iris,wine,breast_cancer")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--seeds", default="20260803")
    parser.add_argument(
        "--optimization-seeds",
        help=(
            "optional comma-separated optimizer seeds; outer folds still use "
            "--seeds, allowing matched-fold multi-seed replication"
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-gen", type=int, default=5)
    parser.add_argument(
        "--pep-iterations",
        type=int,
        help="optional PEP-paper iteration count; default is ceil(n^2 log n)",
    )
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument(
        "--objective-boundary-tolerances",
        default="0.05",
        help=(
            "comma-separated tolerances for da_pddr_obj; multiple values are "
            "run in the same fold with distinct algorithm labels"
        ),
    )
    parser.add_argument("--data-home")
    parser.add_argument("--include-exact", action="store_true")
    parser.add_argument("--exact-max-classifiers", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("results/raw_data") / f"batch_{timestamp}.json"

    config = BatchConfig(
        datasets=_csv_values(args.datasets),
        algorithms=_csv_values(args.algorithms),
        seeds=tuple(int(value) for value in _csv_values(args.seeds)),
        optimization_seeds=(
            tuple(int(value) for value in _csv_values(args.optimization_seeds))
            if args.optimization_seeds
            else ()
        ),
        n_splits=args.n_splits,
        val_size=args.val_size,
        pool_size=args.pool_size,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        pep_iterations=args.pep_iterations,
        timing_repeats=args.timing_repeats,
        objective_boundary_tolerances=tuple(
            float(value)
            for value in _csv_values(args.objective_boundary_tolerances)
        ),
        include_exact_front=args.include_exact,
        exact_max_classifiers=args.exact_max_classifiers,
        data_home=args.data_home,
    )
    records = run_batch(config, output)
    print(f"records = {len(records)}")
    print(f"output = {output}")


if __name__ == "__main__":
    main()
