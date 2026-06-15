from __future__ import annotations

import argparse
import csv
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


COLORS = {
    "v0": "#7f1d1d",
    "v1": "#b45309",
    "v2": "#1d4ed8",
    "v3": "#047857",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot whole-project version sweep results.")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--peak-gflops", type=float, default=89.6)
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=45.0)
    return parser.parse_args()


def rows_by_version(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["version"], []).append(row)
    for version_rows in grouped.values():
        version_rows.sort(key=lambda row: row["n"])
    return grouped


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "version",
        "n",
        "total_ms",
        "estimated_flops",
        "estimated_bytes",
        "arithmetic_intensity",
        "gflops",
        "speedup_vs_v0",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def add_speedups(rows: list[dict]) -> None:
    baseline = {
        row["n"]: row["total_ms"]
        for row in rows
        if row["version"] == "v0"
    }
    for row in rows:
        base_ms = baseline.get(row["n"])
        row["speedup_vs_v0"] = base_ms / row["total_ms"] if base_ms else None


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_runtime(grouped: dict[str, list[dict]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for version, rows in grouped.items():
        ax.plot(
            [row["n"] for row in rows],
            [row["total_ms"] for row in rows],
            marker="o",
            lw=2,
            color=COLORS.get(version),
            label=version,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Scaling factor n (B=64, C=4n, T=16n, heads=32n)")
    ax.set_ylabel("Whole-project runtime [ms], log scale")
    ax.set_title("Whole-project runtime by optimization version")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="version")
    save(fig, out_dir / "project_versions_runtime.svg")
    plot_runtime_png(grouped, out_dir)


def plot_runtime_png(grouped: dict[str, list[dict]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for version, rows in grouped.items():
        ax.plot(
            [row["n"] for row in rows],
            [row["total_ms"] for row in rows],
            marker="o",
            lw=2,
            color=COLORS.get(version),
            label=version,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Scaling factor n (B=64, C=4n, T=16n, heads=32n)")
    ax.set_ylabel("Whole-project runtime [ms], log scale")
    ax.set_title("Whole-project runtime by optimization version")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="version")
    save(fig, out_dir / "project_versions_runtime.png")


def plot_speedup(grouped: dict[str, list[dict]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for version, rows in grouped.items():
        ax.plot(
            [row["n"] for row in rows],
            [row["speedup_vs_v0"] for row in rows],
            marker="o",
            lw=2,
            color=COLORS.get(version),
            label=version,
        )
    ax.axhline(1.0, color="#64748b", lw=1, ls="--")
    ax.set_xlabel("Scaling factor n (B=64, C=4n, T=16n, heads=32n)")
    ax.set_ylabel("Speedup over whole-project v0")
    ax.set_title("Whole-project speedup by optimization version")
    ax.grid(True, alpha=0.25)
    ax.legend(title="version")
    save(fig, out_dir / "project_versions_speedup.svg")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for version, rows in grouped.items():
        ax.plot(
            [row["n"] for row in rows],
            [row["speedup_vs_v0"] for row in rows],
            marker="o",
            lw=2,
            color=COLORS.get(version),
            label=version,
        )
    ax.axhline(1.0, color="#64748b", lw=1, ls="--")
    ax.set_xlabel("Scaling factor n (B=64, C=4n, T=16n, heads=32n)")
    ax.set_ylabel("Speedup over whole-project v0")
    ax.set_title("Whole-project speedup by optimization version")
    ax.grid(True, alpha=0.25)
    ax.legend(title="version")
    save(fig, out_dir / "project_versions_speedup.png")


def plot_roofline(
    grouped: dict[str, list[dict]],
    out_dir: Path,
    peak_gflops: float,
    peak_bandwidth_gbs: float,
) -> None:
    rows = [row for version_rows in grouped.values() for row in version_rows]
    ai_values = [row["arithmetic_intensity"] for row in rows]
    gflops_values = [row["gflops"] for row in rows]
    ai_min = max(min(ai_values) / 3.0, 1e-3)
    ai_max = max(max(ai_values) * 3.0, 1.0)
    perf_min = max(min(gflops_values) / 3.0, 1e-3)
    perf_max = max(max(max(gflops_values) * 3.0, peak_gflops * 1.25), 1.0)

    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 300)
    memory_roof = peak_bandwidth_gbs * ai_grid
    compute_roof = np.full_like(ai_grid, peak_gflops)
    roof = np.minimum(memory_roof, compute_roof)

    for suffix in ("svg", "png"):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.loglog(ai_grid, roof, color="#1f2933", lw=2, label="roofline")
        ax.loglog(ai_grid, memory_roof, "--", color="#64748b", lw=1, label="memory ceiling")
        ax.loglog(ai_grid, compute_roof, "--", color="#94a3b8", lw=1, label="compute ceiling")
        for version, rows in grouped.items():
            ax.plot(
                [row["arithmetic_intensity"] for row in rows],
                [row["gflops"] for row in rows],
                marker="o",
                lw=2,
                color=COLORS.get(version),
                label=version,
            )
            for row in rows:
                ax.annotate(
                    f"n={row['n']}",
                    (row["arithmetic_intensity"], row["gflops"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                )
        ax.set_xlim(ai_min, ai_max)
        ax.set_ylim(perf_min, perf_max)
        ax.set_xlabel("Arithmetic intensity [estimated FLOP/byte]")
        ax.set_ylabel("Performance [estimated GFLOP/s]")
        ax.set_title("Whole-project roofline by optimization version")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        save(fig, out_dir / f"project_versions_roofline.{suffix}")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.json.read_text())
    rows = payload["results"]
    add_speedups(rows)
    grouped = rows_by_version(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "project_versions_summary.csv")
    payload["results"] = rows
    (args.out_dir / "project_versions_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    plot_runtime(grouped, args.out_dir)
    plot_speedup(grouped, args.out_dir)
    plot_roofline(grouped, args.out_dir, args.peak_gflops, args.peak_bandwidth_gbs)
    print(f"wrote plots and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
