from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from roofline_costs import estimate_target_cost

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the benchmark "
        "environment, for example: .venv/bin/python -m pip install matplotlib"
    ) from exc


# Final best-of-each-op configuration. GP is pinned to v2 (not v2_1); attention
# uses ver_3 (newest SIMD QK assembly). gelu has no ver_3, so ver_2 is best.
BEST_CONFIG = [
    "geometric_product_v2",
    "equi_join_v3",
    "equi_linear_ver_3",
    "equi_rms_norm_ver_3",
    "scaler_gated_gelu_ver_2",
    "equi_geometric_attention_ver_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final best-config whole-project roofline across n (single curve)."
    )
    parser.add_argument(
        "--runtime-json",
        action="append",
        type=Path,
        required=True,
        help="benchmark_repo JSON output. Pass once per n value.",
    )
    parser.add_argument("--peak-gflops", type=float, required=True)
    parser.add_argument("--peak-bandwidth-gbs", type=float, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--title", default="Whole-project final roofline (best config) by problem size n")
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--dim-formula", choices=["old", "new"], default="new")
    return parser.parse_args()


def aggregate(payload: dict, dtype_bytes: int, formula: str = "new") -> dict:
    n = int(payload["n"])
    rows_by_target = {row["target"]: row for row in payload["results"]}

    missing = [t for t in BEST_CONFIG if t not in rows_by_target]
    if missing:
        raise SystemExit(f"Missing targets for n={n}: {', '.join(missing)}")

    total_ms = 0.0
    total_flops = 0.0
    total_bytes = 0.0
    for target in BEST_CONFIG:
        row = rows_by_target[target]
        cost = estimate_target_cost(target, n, dtype_bytes=dtype_bytes, formula=formula)
        if cost is None:
            raise SystemExit(f"No cost model for {target}")
        total_ms += float(row["mean_ms"])
        total_flops += float(cost["estimated_flops_per_call"])
        total_bytes += float(cost["estimated_bytes_per_call"])

    return {
        "config": "final-best",
        "n": n,
        "targets": BEST_CONFIG,
        "total_ms": total_ms,
        "estimated_flops": total_flops,
        "estimated_bytes": total_bytes,
        "arithmetic_intensity": total_flops / total_bytes,
        "gflops": total_flops / (total_ms / 1000.0) / 1e9,
        "cost_source": "analytical sum",
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "config", "n", "total_ms", "estimated_flops", "estimated_bytes",
        "arithmetic_intensity", "gflops", "cost_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def plot(rows: list[dict], peak_gflops: float, peak_bandwidth_gbs: float, out_path: Path, title: str) -> None:
    ai_values = [float(r["arithmetic_intensity"]) for r in rows]
    gflops_values = [float(r["gflops"]) for r in rows]
    ai_min = max(min(ai_values) / 3.0, 1e-3)
    ai_max = max(max(ai_values) * 3.0, 1.0)
    perf_min = max(min(gflops_values) / 3.0, 1e-3)
    perf_max = max(max(max(gflops_values) * 3.0, peak_gflops * 1.25), 1.0)

    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 300)
    memory_roof = peak_bandwidth_gbs * ai_grid
    compute_roof = np.full_like(ai_grid, peak_gflops)
    roof = np.minimum(memory_roof, compute_roof)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(ai_grid, roof, color="#1f2933", lw=2.0, label="roofline")
    ax.loglog(ai_grid, memory_roof, "--", color="#64748b", lw=1.0, label="memory ceiling")
    ax.loglog(ai_grid, compute_roof, "--", color="#94a3b8", lw=1.0, label="compute ceiling")

    ordered = sorted(rows, key=lambda r: r["n"])
    xs = [r["arithmetic_intensity"] for r in ordered]
    ys = [r["gflops"] for r in ordered]
    ax.plot(xs, ys, marker="o", lw=2.0, ms=7, color="#047857", label="final best config")
    for r in ordered:
        ax.annotate(
            f"n={r['n']}",
            (r["arithmetic_intensity"], r["gflops"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(perf_min, perf_max)
    ax.set_xlabel("Arithmetic intensity [estimated FLOP/byte]")
    ax.set_ylabel("Performance [estimated GFLOP/s]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.runtime_json]
    rows = [aggregate(p, args.dtype_bytes, formula=args.dim_formula) for p in sorted(payloads, key=lambda item: int(item["n"]))]

    output = {
        "description": "Final best-config whole-project roofline (measured runtime, analytical FLOP/byte cost model). GP pinned to v2, attention ver_3.",
        "peak_gflops": args.peak_gflops,
        "peak_bandwidth_gbs": args.peak_bandwidth_gbs,
        "best_config": BEST_CONFIG,
        "results": rows,
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    write_csv(rows, args.out_prefix.with_suffix(".csv"))
    plot(rows, args.peak_gflops, args.peak_bandwidth_gbs, args.out_prefix.with_suffix(".svg"), args.title)
    plot(rows, args.peak_gflops, args.peak_bandwidth_gbs, args.out_prefix.with_suffix(".png"), args.title)
    for ext in ("json", "csv", "svg", "png"):
        print(f"wrote {args.out_prefix.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
