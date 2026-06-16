"""Regenerate the input JSON that plot_runtime.py and plot_roofline.py expect.

Uses the new dimension formula B=2n, H=4n, T=128n, C=8 and sweeps n=1..12.
Attention v0/v1 cap at n=3 (too slow beyond that), v2 at n=5, v3 at n=9.

    python benchmarks/gen_legacy_plot_data.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_repo as br  # noqa: E402

torch.set_num_threads(1)
device = torch.device("cpu")

NON_ATTN = [
    "geometric_product_v0", "geometric_product_v1", "geometric_product_v2", "geometric_product_v3",
    "equi_join_v0", "equi_join_v1", "equi_join_v2", "equi_join_v3",
    "equi_linear_ver_0", "equi_linear_ver_1", "equi_linear_ver_2", "equi_linear_ver_3",
    "equi_rms_norm_ver_0", "equi_rms_norm_ver_1", "equi_rms_norm_ver_2", "equi_rms_norm_ver_3",
    "scaler_gated_gelu_ver_0", "scaler_gated_gelu_ver_1", "scaler_gated_gelu_ver_2", "scaler_gated_gelu_ver_3",
]
ATTN = [
    "equi_geometric_attention_ver_0", "equi_geometric_attention_ver_1",
    "equi_geometric_attention_ver_2", "equi_geometric_attention_ver_3",
]
# Attention caps: v0/v1 feasible to n=5 (T=640, ~51s/call turbo-off), v2 same, v3 to n=9
ATTN_MAX_N = {
    "equi_geometric_attention_ver_0": 5,
    "equi_geometric_attention_ver_1": 5,
    "equi_geometric_attention_ver_2": 5,
    "equi_geometric_attention_ver_3": 9,
}

NS = list(range(1, 13))  # n=1..12

_orig_make_inputs = br.make_inputs
_cache: dict[int, dict] = {}
def _cached_make_inputs(dev, n):
    if n not in _cache:
        _cache[n] = _orig_make_inputs(dev, n)
    return _cache[n]
br.make_inputs = _cached_make_inputs


def measure(target, n, warmup, repeats, inner):
    row = br.measure_target(target, device=device, n=n,
                            warmup=warmup, repeats=repeats, inner_iters=inner)
    return row["min_ms"]


def main():
    results: dict[int, dict[str, float]] = {n: {} for n in NS}
    for n in NS:
        print(f"\n--- n={n} cfg={br.make_cfg(n)} ---", flush=True)
        for t in NON_ATTN:
            results[n][t] = measure(t, n, warmup=1, repeats=3, inner=3)
            print(f"  {t:34s} {results[n][t]:10.3f} ms", flush=True)
        for t in ATTN:
            if n > ATTN_MAX_N[t]:
                print(f"  {t:34s}   [skipped n>{ATTN_MAX_N[t]}]", flush=True)
                continue
            w, r, i = (1, 3, 1)
            results[n][t] = measure(t, n, warmup=w, repeats=r, inner=i)
            print(f"  {t:34s} {results[n][t]:10.3f} ms", flush=True)
        _cache.pop(n, None)

    base = Path("benchmarks/results")
    (base / "project_versions").mkdir(parents=True, exist_ok=True)
    (base / "run_2026_06_02").mkdir(parents=True, exist_ok=True)

    # project_versions: nested format expected by plot_runtime.py / plot_roofline.py
    proj = {"device": "cpu", "threads": 1, "dim_formula": "B=2n,H=4n,T=128n,C=8",
            "results": [
        {"n": n, "targets": [{"target": t, "min_ms": results[n][t]}
                              for t in (NON_ATTN + ATTN) if t in results[n]]}
        for n in NS
    ]}
    (base / "project_versions" / "project_versions_n1_n12_runtime.json").write_text(json.dumps(proj, indent=2))

    # Per-n attention files (flat {results:[{target,min_ms}]})
    for n in NS:
        attn_targets = [t for t in ATTN if t in results[n]]
        if not attn_targets:
            continue
        payload = {"n": n, "results": [{"target": t, "min_ms": results[n][t]} for t in attn_targets]}
        (base / "run_2026_06_02" / f"attention_n{n}.json").write_text(json.dumps(payload, indent=2))

    print("\nwrote project_versions + run_2026_06_02 JSON")


if __name__ == "__main__":
    main()
