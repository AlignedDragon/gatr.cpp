from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_repo as br  # noqa: E402
from roofline_costs import estimate_target_cost  # noqa: E402


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
    "v3_1": [
        "geometric_product_v3",
        "equi_join_v3",
        "equi_linear_ver_3",
        "equi_rms_norm_ver_3",
        "scaler_gated_gelu_ver_3",
        "equi_geometric_attention_ver_3_1",
    ],
}

# Attention is O(T^2); v0/v1 feasible to n=5 (T=640, ~51s/call), v2 same, v3 to n=9
ATTN_MAX_N = {"v0": 5, "v1": 5, "v2": 5, "v3": 9, "v3_1": 9}
ATTN_TARGETS = {t for ts in VERSION_TARGETS.values() for t in ts if "attention" in t}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whole-project v0/v1/v2/v3 runtime across the TA n scaling."
    )
    parser.add_argument("--n", type=int, action="append", required=True)
    parser.add_argument("--version", choices=sorted(VERSION_TARGETS), action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner-iters", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_one(
    target_name: str,
    device: torch.device,
    n: int,
    warmup: int,
    repeats: int,
    inner_iters: int,
) -> dict:
    fn = br.build_target(target_name, device=device, n=n)
    with torch.inference_mode():
        for _ in range(warmup):
            for _ in range(inner_iters):
                fn()
        synchronize(device)

        times_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(inner_iters):
                fn()
            synchronize(device)
            times_ms.append((time.perf_counter() - start) * 1e3 / inner_iters)

    return {
        "target": target_name,
        "n": n,
        "mean_ms": statistics.fmean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def aggregate_version(
    version: str,
    n: int,
    target_rows: list[dict],
    dtype_bytes: int,
) -> dict:
    total_ms = sum(float(row["mean_ms"]) for row in target_rows)
    total_flops = 0.0
    total_bytes = 0.0
    for row in target_rows:
        cost = estimate_target_cost(str(row["target"]), n, dtype_bytes=dtype_bytes, formula="new")
        if cost is None:
            raise SystemExit(f"No cost model for {row['target']}")
        total_flops += float(cost["estimated_flops_per_call"])
        total_bytes += float(cost["estimated_bytes_per_call"])

    return {
        "version": version,
        "n": n,
        "cfg": br.make_cfg(n),
        "total_ms": total_ms,
        "estimated_flops": total_flops,
        "estimated_bytes": total_bytes,
        "arithmetic_intensity": total_flops / total_bytes,
        "gflops": total_flops / (total_ms / 1000.0) / 1e9,
        "targets": target_rows,
    }


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def print_row(row: dict) -> None:
    print(
        f"{row['version']:>2s} n={row['n']:<2d} "
        f"{row['total_ms']:10.3f} ms "
        f"{row['gflops']:9.3f} GF/s "
        f"{row['arithmetic_intensity']:8.3f} FLOP/B"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    versions = args.version or list(VERSION_TARGETS)
    ns = sorted(dict.fromkeys(args.n))
    payload = {
        "description": "Whole-project version runtime sweep with scaling B=2n, H=4n, T=128n, C=8.",
        "device": str(device),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "inner_iters": args.inner_iters,
        "dtype_bytes": args.dtype_bytes,
        "version_targets": {version: VERSION_TARGETS[version] for version in versions},
        "results": [],
    }
    checkpoint(args.json_out, payload)

    print(f"{'ver':>2s} {'n':>3s} {'total_ms':>13s} {'GF/s':>14s} {'FLOP/B':>13s}")
    for n in ns:
        for version in versions:
            target_rows = []
            for target in VERSION_TARGETS[version]:
                if target in ATTN_TARGETS and n > ATTN_MAX_N[version]:
                    print(f"  skipping {target} for {version} n={n} (attention cap)")
                    continue
                row = measure_one(
                    target_name=target,
                    device=device,
                    n=n,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    inner_iters=args.inner_iters,
                )
                target_rows.append(row)
            aggregate = aggregate_version(version, n, target_rows, args.dtype_bytes)
            payload["results"].append(aggregate)
            checkpoint(args.json_out, payload)
            print_row(aggregate)

    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
