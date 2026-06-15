"""Convert per_function.json (run_full_bench.py format, B=2n/H=4n/T=128n/C=8)
to one benchmark_repo-style JSON per n value, suitable for
plot_project_versions_roofline.py and plot_final_best_roofline.py.

Usage:
    python benchmarks/convert_perfunc_to_bench_repo.py \
        --json benchmarks/results/run_20260614/per_function.json \
        --out-dir benchmarks/results/bench_repo_new_dims
"""
import argparse
import json
from pathlib import Path

POINTWISE_TARGET_MAP = {
    "geometric_product": {"v0": "geometric_product_v0", "v1": "geometric_product_v1",
                          "v2": "geometric_product_v2", "v3": "geometric_product_v3"},
    "equi_join":         {"v0": "equi_join_v0", "v1": "equi_join_v1",
                          "v2": "equi_join_v2", "v3": "equi_join_v3"},
    "equi_linear":       {"v0": "equi_linear_ver_0", "v1": "equi_linear_ver_1",
                          "v2": "equi_linear_ver_2", "v3": "equi_linear_ver_3"},
    "equi_rms_norm":     {"v0": "equi_rms_norm_ver_0", "v1": "equi_rms_norm_ver_1",
                          "v2": "equi_rms_norm_ver_2", "v3": "equi_rms_norm_ver_3"},
    "scaler_gated_gelu": {"v0": "scaler_gated_gelu_ver_0", "v1": "scaler_gated_gelu_ver_1",
                          "v2": "scaler_gated_gelu_ver_2", "v3": "scaler_gated_gelu_ver_3"},
}
ATTENTION_TARGET_MAP = {
    "v0": "equi_geometric_attention_ver_0",
    "v1": "equi_geometric_attention_ver_1",
    "v2": "equi_geometric_attention_ver_2",
    "v3": "equi_geometric_attention_ver_3",
    "v3_1": "equi_geometric_attention_ver_3",  # v3_1 is the canonical v3
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all n values across pointwise and attention
    pw = data["pointwise"]
    att = data["attention"]

    all_ns = set(pw["geometric_product"]["n_values"])
    out_by_n: dict[int, dict] = {}

    # Pointwise ops
    for op, op_data in pw.items():
        ns = op_data["n_values"]
        for ver, timings in op_data["versions"].items():
            if ver == "py":
                continue
            target = POINTWISE_TARGET_MAP[op].get(ver)
            if target is None:
                continue
            for i, n in enumerate(ns):
                if n not in out_by_n:
                    out_by_n[n] = {}
                out_by_n[n][target] = timings[i]

    # Attention
    for ver, n_data in att.items():
        target = ATTENTION_TARGET_MAP.get(ver)
        if target is None:
            continue
        for n_str, min_ms in n_data.items():
            n = int(n_str)
            if n not in out_by_n:
                out_by_n[n] = {}
            # v3_1 is better than v3; prefer it for the ver_3 slot
            if ver == "v3_1" or target not in out_by_n[n]:
                out_by_n[n][target] = min_ms

    # Write one file per n
    for n, targets in sorted(out_by_n.items()):
        cfg = {"batch": 2 * n, "tokens": 128 * n, "channels": 8, "heads": 4 * n}
        results = [
            {"target": t, "n": n, "mean_ms": ms, "p50_ms": ms, "min_ms": ms, "max_ms": ms}
            for t, ms in targets.items()
        ]
        payload = {"n": n, "cfg": cfg, "device": "cpu", "threads": 1,
                   "warmup": 1, "repeats": 1, "inner_iters": 1, "results": results}
        out = args.out_dir / f"bench_repo_n{n}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}  ({len(results)} targets)")


if __name__ == "__main__":
    main()
