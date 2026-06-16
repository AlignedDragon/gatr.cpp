from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ── PAPI n=3 reference + Python timings ────────────────────────────────────────
def _load_papi3(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for r in d["results"]:
        fl = r.get("flops_per_call"); by = r.get("memory_bytes_per_call")
        if fl and by:
            out[r["target"]] = {"flops": fl, "bytes": by}
    return out

_papi3 = _load_papi3("benchmarks/results/papi_1thread_n3.json")

_pf_path = Path("benchmarks/results/run_20260615/per_function.json")
_py_pw: dict[str, dict[int, float]] = {}
_py_attn: dict[int, float] = {}
if _pf_path.exists():
    _pf = json.loads(_pf_path.read_text())
    for op, dat in _pf["pointwise"].items():
        _py_pw[op] = {n: dat["versions"]["py"][i] for i, n in enumerate(dat["n_values"])}
    _py_attn = {int(k): float(v) for k, v in _pf["attention"]["py"].items()}

_N_REF = 3
_NA_FL_EXP, _NA_BY_EXP     = 2.0, 5.5
_ATTN_FL_EXP, _ATTN_BY_EXP = 4.0, 3.29
_PY_REFS = [
    ("geometric_product",  "geometric_product_v0"),
    ("equi_join",          "equi_join_v0"),
    ("equi_linear",        "equi_linear_ver_0"),
    ("equi_rms_norm",      "equi_rms_norm_ver_0"),
    ("scaler_gated_gelu",  "scaler_gated_gelu_ver_0"),
]


def _sweep_py_best(n: int):
    """Python whole-project point: PAPI-scaled FLOPs + Python timing."""
    if not _papi3 or not _py_pw:
        return None
    total_fl = total_by = total_ms = 0.0
    for py_key, papi_ref in _PY_REFS:
        ref = _papi3.get(papi_ref)
        t = _py_pw.get(py_key, {}).get(n)
        if not ref or t is None:
            return None
        total_fl += ref["flops"] * (n / _N_REF) ** _NA_FL_EXP
        total_by += ref["bytes"] * (n / _N_REF) ** _NA_BY_EXP
        total_ms += t
    t_attn = _py_attn.get(n)
    if t_attn is not None:
        ref = _papi3.get("equi_geometric_attention_ver_3")
        if ref:
            total_fl += ref["flops"] * (n / _N_REF) ** _ATTN_FL_EXP
            total_by += ref["bytes"] * (n / _N_REF) ** _ATTN_BY_EXP
            total_ms += t_attn
    if not total_by or not total_ms:
        return None
    return total_fl / total_by, total_fl / (total_ms / 1000) / 1e9

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
    "equi_geometric_attention_ver_3_1",
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


_ATTN_TARGETS = {"equi_geometric_attention_ver_3_1"}
_ATTN_PAPI_REF = "equi_geometric_attention_ver_3"  # ver_3_1 uses ver_3 PAPI data


def aggregate(payload: dict, dtype_bytes: int, formula: str = "new") -> dict:
    n = int(payload["n"])
    rows_by_target = {row["target"]: row for row in payload["results"]}

    missing = [t for t in BEST_CONFIG if t not in rows_by_target]
    if missing:
        raise SystemExit(f"Missing targets for n={n}: {', '.join(missing)}")

    if not _papi3:
        raise SystemExit("PAPI reference data not found — expected benchmarks/results/papi_1thread_n3.json")

    total_ms = 0.0
    total_flops = 0.0
    total_bytes = 0.0
    for target in BEST_CONFIG:
        row = rows_by_target[target]
        if target in _ATTN_TARGETS:
            papi_key = _ATTN_PAPI_REF
            fl_exp, by_exp = _ATTN_FL_EXP, _ATTN_BY_EXP
        else:
            papi_key = target
            fl_exp, by_exp = _NA_FL_EXP, _NA_BY_EXP
        ref = _papi3.get(papi_key)
        if ref is None:
            raise SystemExit(f"No PAPI reference for {papi_key}")
        total_ms += float(row["mean_ms"])
        total_flops += ref["flops"] * (n / _N_REF) ** fl_exp
        total_bytes += ref["bytes"] * (n / _N_REF) ** by_exp

    return {
        "config": "final-best",
        "n": n,
        "targets": BEST_CONFIG,
        "total_ms": total_ms,
        "estimated_flops": total_flops,
        "estimated_bytes": total_bytes,
        "arithmetic_intensity": total_flops / total_bytes,
        "gflops": total_flops / (total_ms / 1000.0) / 1e9,
        "cost_source": "PAPI-anchored",
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


def plot(rows: list[dict], peak_gflops: float, peak_bandwidth_gbs: float, out_path: Path, title: str,
         py_curve: list[tuple] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    # Full roofline — same style as all other roofline plots
    ai_grid = np.logspace(-2, 4, 500)
    roof = np.minimum(peak_bandwidth_gbs * ai_grid, peak_gflops)
    ax.plot(ai_grid, roof, 'k-', lw=2.5, zorder=5)
    ridge = peak_gflops / peak_bandwidth_gbs
    ax.axvline(ridge, color='k', ls='--', lw=1, alpha=0.35)
    ax.text(0.13, peak_bandwidth_gbs * 0.13 * 1.3, f'Mem BW: {peak_bandwidth_gbs} GB/s',
            fontsize=8.5, rotation=36, va='bottom')
    ax.text(3500, peak_gflops * 1.08, f'Peak: {peak_gflops} GFLOPS', fontsize=8.5, ha='right')
    ax.text(ridge * 1.1, peak_gflops * 0.45, f'Ridge\n{ridge:.1f} F/B', fontsize=7.5, color='gray')

    ordered = sorted(rows, key=lambda r: r["n"])
    xs = [r["arithmetic_intensity"] for r in ordered]
    ys = [r["gflops"] for r in ordered]
    ax.plot(xs, ys, marker="o", lw=2.0, ms=7, color="#047857", label="final best config")
    label_ns = {1, 3, 6, 9}  # label only a few to avoid crowding
    for r in ordered:
        if r["n"] in label_ns:
            ax.annotate(
                f"n={r['n']}",
                (r["arithmetic_intensity"], r["gflops"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
            )

    # Python dashed reference curve
    if py_curve:
        py_xs = [p[0] for p in py_curve]; py_ys = [p[1] for p in py_curve]
        ax.plot(py_xs, py_ys, "--", color="#777777", lw=1.8, alpha=0.85, zorder=5,
                label="Python (PyTorch)")
        ax.scatter(py_xs, py_ys, s=90, marker="*", color="#777777", zorder=7)
        ai_last, gf_last, n_last = py_curve[-1]
        ax.annotate(f"Python\n(n=1..{n_last})", (ai_last, gf_last),
                    textcoords="offset points", xytext=(6, -16), fontsize=8, color="#555")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.01, 8000)
    ax.set_ylim(0.01, peak_gflops * 2)
    ax.set_xlabel("Arithmetic Intensity [PAPI-anchored FLOP/byte]", fontsize=12)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=12)
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
        "description": "Final best-config whole-project roofline (measured runtime, PAPI-anchored FLOP/byte). GP pinned to v2, attention ver_3_1.",
        "peak_gflops": args.peak_gflops,
        "peak_bandwidth_gbs": args.peak_bandwidth_gbs,
        "best_config": BEST_CONFIG,
        "results": rows,
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Python reference curve
    py_curve = []
    for _n in range(1, 5):
        _pt = _sweep_py_best(_n)
        if _pt:
            py_curve.append((_pt[0], _pt[1], _n))

    args.out_prefix.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    write_csv(rows, args.out_prefix.with_suffix(".csv"))
    plot(rows, args.peak_gflops, args.peak_bandwidth_gbs, args.out_prefix.with_suffix(".svg"), args.title,
         py_curve=py_curve)
    plot(rows, args.peak_gflops, args.peak_bandwidth_gbs, args.out_prefix.with_suffix(".png"), args.title,
         py_curve=py_curve)
    for ext in ("json", "csv", "svg", "png"):
        print(f"wrote {args.out_prefix.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
