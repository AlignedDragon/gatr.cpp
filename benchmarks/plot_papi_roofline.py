from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the benchmark "
        "environment, for example: .venv/bin/python -m pip install matplotlib"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Plot roofline data from run_papi_roofline.py JSON.")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--peak-gflops", type=float, required=True)
    parser.add_argument("--peak-bandwidth-gbs", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Project roofline")
    parser.add_argument("--target", action="append", default=None)
    return parser.parse_args()


def row_points(payload: dict, targets: list[str] | None):
    rows = payload["results"]
    if targets is not None:
        selected = set(targets)
        rows = [row for row in rows if row["target"] in selected]

    points = []
    for row in rows:
        ai = row.get("arithmetic_intensity")
        gflops = row.get("gflops")
        if ai is None or gflops is None or ai <= 0 or gflops <= 0:
            continue
        points.append((row["target"], float(ai), float(gflops), row.get("cost_source", "papi")))
    return points


def main() -> None:
    args = parse_args()
    payload = json.loads(args.json.read_text())
    points = row_points(payload, args.target)
    if not points:
        raise SystemExit("No rows with arithmetic_intensity and gflops were found.")

    ai_values = [p[1] for p in points]
    ai_min = max(min(ai_values) / 4, 1e-3)
    ai_max = max(max(ai_values) * 4, 1.0)
    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 300)
    memory_roof = args.peak_bandwidth_gbs * ai_grid
    compute_roof = np.full_like(ai_grid, args.peak_gflops)
    roof = np.minimum(memory_roof, compute_roof)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ai_grid, roof, color="#222", lw=1.8, label="roofline")
    ax.loglog(ai_grid, memory_roof, "--", color="#777", lw=1.0, label="memory ceiling")
    ax.loglog(ai_grid, compute_roof, "--", color="#999", lw=1.0, label="compute ceiling")

    cmap = plt.get_cmap("tab20")
    for i, (target, ai, gflops, source) in enumerate(points):
        ax.scatter(ai, gflops, s=60, color=cmap(i % 20), label=f"{target} ({source})")
        ax.annotate(target, (ai, gflops), textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
