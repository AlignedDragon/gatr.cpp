"""Regenerate the input JSON that plot_runtime.py and plot_roofline.py expect.

Those scripts read a multi-n aggregate (project_versions/) plus per-n attention /
non-attention files (run_2026_06_02/) whose data was deleted. This driver
recreates all of it from the committed benchmark_repo timing code:

  results/project_versions/project_versions_n1_n3_runtime.json   (nested by n)
  results/run_2026_06_02/attention_n{1,2,3,4}.json
  results/run_2026_06_02/non_attention_n4.json

Inputs are cached per n (benchmark_repo.make_inputs otherwise re-allocates the
~6 GB attention tensors for every target at n=4). Attention uses few iterations
because attention v0 at n=4 is ~250x slower than at n=1.

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

# Cache inputs per n so the heavy attention tensors are allocated once, not per target.
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
    results: dict[int, dict[str, float]] = {n: {} for n in (1, 2, 3, 4)}
    for n in (1, 2, 3, 4):
        print(f"\n--- n={n} (cfg={br.make_cfg(n)}) ---", flush=True)
        for t in NON_ATTN:
            results[n][t] = measure(t, n, warmup=2, repeats=8, inner=8)
            print(f"  {t:34s} {results[n][t]:10.3f} ms", flush=True)
        # Attention: cheaper iteration budget; v0/v1 at n>=3 are very slow.
        w, r, i = (2, 5, 3) if n <= 2 else (1, 3, 1)
        for t in ATTN:
            results[n][t] = measure(t, n, warmup=w, repeats=r, inner=i)
            print(f"  {t:34s} {results[n][t]:10.3f} ms", flush=True)
        _cache.pop(n, None)  # free this n's inputs before the next (bigger) n

    base = Path("benchmarks/results")
    (base / "project_versions").mkdir(parents=True, exist_ok=True)
    (base / "run_2026_06_02").mkdir(parents=True, exist_ok=True)

    # project_versions: nested {results:[{n, targets:[{target,min_ms}]}]} for n=1,2,3.
    proj = {"device": "cpu", "threads": 1, "results": [
        {"n": n, "targets": [{"target": t, "min_ms": results[n][t]} for t in (NON_ATTN + ATTN)]}
        for n in (1, 2, 3)
    ]}
    (base / "project_versions" / "project_versions_n1_n3_runtime.json").write_text(json.dumps(proj, indent=2))

    # Per-n attention + n=4 non-attention, flat {results:[{target,min_ms}]}.
    for n in (1, 2, 3, 4):
        payload = {"n": n, "results": [{"target": t, "min_ms": results[n][t]} for t in ATTN]}
        (base / "run_2026_06_02" / f"attention_n{n}.json").write_text(json.dumps(payload, indent=2))
    n4_non = {"n": 4, "results": [{"target": t, "min_ms": results[4][t]} for t in NON_ATTN]}
    (base / "run_2026_06_02" / "non_attention_n4.json").write_text(json.dumps(n4_non, indent=2))

    print("\nwrote project_versions + run_2026_06_02 JSON")


if __name__ == "__main__":
    main()
