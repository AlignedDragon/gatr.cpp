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


VERSION_TARGETS = {
    "v0": [
        "geometric_product_v0",
        "equi_join_v0",
        "equi_linear_ver_0",
        "equi_rms_norm_ver_0",
        "scaler_gated_gelu_ver_0",
        "equi_geometric_attention_ver_0",
    ],
    "v1": [
        "geometric_product_v1",
        "equi_join_v1",
        "equi_linear_ver_1",
        "equi_rms_norm_ver_1",
        "scaler_gated_gelu_ver_1",
        "equi_geometric_attention_ver_1",
    ],
    "v2": [
        "geometric_product_v2",
        "equi_join_v2",
        "equi_linear_ver_2",
        "equi_rms_norm_ver_2",
        "scaler_gated_gelu_ver_2",
        "equi_geometric_attention_ver_2",
    ],
    "v3": [
        "geometric_product_v3",
        "equi_join_v3",
        "equi_linear_ver_3",
        "equi_rms_norm_ver_3",
        "scaler_gated_gelu_ver_3",
        "equi_geometric_attention_ver_3",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark_repo JSON files into project-version roofline curves."
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
    parser.add_argument("--title", default="Whole-project roofline by optimization version")
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--dim-formula", choices=["old", "new"], default="new")
    return parser.parse_args()


def aggregate_version(payload: dict, version: str, targets: list[str], dtype_bytes: int, formula: str = "new") -> dict | None:
    n = int(payload["n"])
    rows_by_target = {row["target"]: row for row in payload["results"]}

    missing = [target for target in targets if target not in rows_by_target]
    if missing:
        return None  # this version has no data at this n (e.g. attention cap)

    total_ms = 0.0
    total_flops = 0.0
    total_bytes = 0.0
    cost_models: list[str] = []

    for target in targets:
        row = rows_by_target[target]
        cost = estimate_target_cost(target, n, dtype_bytes=dtype_bytes, formula=formula)
        if cost is None:
            raise SystemExit(f"No cost model for {target}")

        total_ms += float(row["mean_ms"])
        total_flops += float(cost["estimated_flops_per_call"])
        total_bytes += float(cost["estimated_bytes_per_call"])
        cost_models.append(str(cost["cost_model"]))

    total_seconds = total_ms / 1000.0
    return {
        "version": version,
        "n": n,
        "targets": targets,
        "total_ms": total_ms,
        "estimated_flops": total_flops,
        "estimated_bytes": total_bytes,
        "arithmetic_intensity": total_flops / total_bytes,
        "gflops": total_flops / total_seconds / 1e9,
        "cost_source": "analytical sum",
        "cost_models": sorted(set(cost_models)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "version",
        "n",
        "total_ms",
        "estimated_flops",
        "estimated_bytes",
        "arithmetic_intensity",
        "gflops",
        "cost_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def draw_roofline(ax, peak_gflops, peak_bandwidth_gbs):
    ai_range = np.logspace(-1, 4, 500)
    roof = np.minimum(peak_bandwidth_gbs * ai_range, peak_gflops)
    ax.plot(ai_range, roof, 'k-', lw=2.5, zorder=5)
    ax.axvline(peak_gflops / peak_bandwidth_gbs, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.13, peak_bandwidth_gbs * 0.13 * 1.3, f'Mem BW: {peak_bandwidth_gbs} GB/s',
            fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, peak_gflops * 1.08, f'Peak: {peak_gflops} GFLOPS', fontsize=8.5, ha='right')


def plot(rows: list[dict], peak_gflops: float, peak_bandwidth_gbs: float, out_path: Path, title: str) -> None:
    import matplotlib.lines as mlines

    ver_color  = {"v0": "#B0BEC5", "v1": "#FF9800", "v2": "#2196F3", "v3": "#4CAF50"}
    ver_marker = {"v0": "o", "v1": "s", "v2": "^", "v3": "D"}
    ver_size   = {"v0": 60, "v1": 75, "v2": 95, "v3": 115}
    ver_alpha  = {"v0": 0.55, "v1": 0.70, "v2": 0.85, "v3": 1.0}
    ver_label  = {"v0": "v0 — Baseline", "v1": "v1 — Math", "v2": "v2 — Scalar mem.", "v3": "v3 — SIMD"}

    # Group rows by n
    by_n: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_n.setdefault(row["n"], {})[row["version"]] = row

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_roofline(ax, peak_gflops, peak_bandwidth_gbs)

    # Draw one scatter point per (version, n); arrows v0→v3 per n
    for n, ver_dict in sorted(by_n.items()):
        # Arrows v0→v3 between adjacent versions
        vers_present = [v for v in ("v0", "v1", "v2", "v3") if v in ver_dict]
        for va, vb in zip(vers_present, vers_present[1:]):
            ax.annotate(
                '', xy=(ver_dict[vb]["arithmetic_intensity"], ver_dict[vb]["gflops"]),
                xytext=(ver_dict[va]["arithmetic_intensity"], ver_dict[va]["gflops"]),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.1, alpha=0.5),
            )

        for ver, row in ver_dict.items():
            ai, gf = row["arithmetic_intensity"], row["gflops"]
            ax.scatter(ai, gf,
                       s=ver_size[ver], marker=ver_marker[ver],
                       color=ver_color[ver], alpha=ver_alpha[ver], zorder=10,
                       edgecolors=ver_color[ver], linewidths=0.8)

        # Label n on v3 (or best available version)
        best = ver_dict.get("v3") or ver_dict.get("v2") or ver_dict.get("v1") or ver_dict.get("v0")
        if best:
            ax.annotate(f'n={n}', (best["arithmetic_intensity"], best["gflops"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color='#2E7D32', fontweight='bold')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, 8000); ax.set_ylim(0.1, peak_gflops * 2)
    ax.set_xlabel('Arithmetic Intensity [estimated FLOP/byte]', fontsize=12)
    ax.set_ylabel('Performance [estimated GFLOP/s]', fontsize=12)
    ax.set_title(title + '\nB=2n H=4n T=128n C=8  |  Tiger Lake i7-1165G7  |  Arrows: v0→v3', fontsize=11)
    ax.grid(True, which='both', alpha=0.2, ls='--')

    leg = [mlines.Line2D([], [], color=ver_color[v], marker=ver_marker[v],
                         ls='None', ms=8, alpha=ver_alpha[v], label=ver_label[v])
           for v in ("v0", "v1", "v2", "v3")]
    ax.legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.runtime_json]

    rows: list[dict] = []
    for payload in sorted(payloads, key=lambda item: int(item["n"])):
        for version, targets in VERSION_TARGETS.items():
            row = aggregate_version(payload, version, targets, args.dtype_bytes, formula=args.dim_formula)
            if row is not None:
                rows.append(row)

    output = {
        "description": "Whole-project roofline aggregation from benchmark_repo runtimes and analytical cost estimates.",
        "peak_gflops": args.peak_gflops,
        "peak_bandwidth_gbs": args.peak_bandwidth_gbs,
        "version_targets": VERSION_TARGETS,
        "results": rows,
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    csv_path = args.out_prefix.with_suffix(".csv")
    svg_path = args.out_prefix.with_suffix(".svg")
    png_path = args.out_prefix.with_suffix(".png")

    json_path.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(rows, csv_path)
    plot(rows, args.peak_gflops, args.peak_bandwidth_gbs, svg_path, args.title)
    plot(rows, args.peak_gflops, args.peak_bandwidth_gbs, png_path, args.title)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
