from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_repo as br  # noqa: E402
from roofline_costs import estimate_target_cost  # noqa: E402


DEFAULT_EVENTS = [
    "PAPI_TOT_CYC",
    "PAPI_TOT_INS",
    "PAPI_FP_OPS",
    "PAPI_L3_TCM",
]

FLOP_EVENTS = ("PAPI_FP_OPS", "PAPI_SP_OPS", "PAPI_DP_OPS")
MEMORY_EVENTS = ("PAPI_L3_TCM",)


def load_papi_events(event_names: list[str]):
    try:
        from pypapi import events as papi_events
        from pypapi import papi_high
    except ImportError as exc:
        raise SystemExit(
            "pypapi is not installed. Install PAPI and pypapi on the benchmark "
            "machine, then rerun this script. Example: module load papi; "
            "python -m pip install pypapi"
        ) from exc

    event_codes = []
    resolved_names = []
    for name in event_names:
        try:
            event_codes.append(getattr(papi_events, name))
            resolved_names.append(name)
        except AttributeError as exc:
            raise SystemExit(
                f"PAPI event {name!r} is not available in pypapi.events. "
                "Use --event to choose events exposed by this machine."
            ) from exc

    return papi_high, resolved_names, event_codes


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measured_flops(counters: dict[str, int], inner_iters: int) -> float | None:
    if "PAPI_FP_OPS" in counters:
        return counters["PAPI_FP_OPS"] / inner_iters

    total = 0
    used = False
    for name in ("PAPI_SP_OPS", "PAPI_DP_OPS"):
        if name in counters:
            total += counters[name]
            used = True
    return total / inner_iters if used else None


def measured_memory_bytes(
    counters: dict[str, int],
    inner_iters: int,
    cache_line_bytes: int,
) -> float | None:
    if "PAPI_L3_TCM" in counters:
        return counters["PAPI_L3_TCM"] * cache_line_bytes / inner_iters
    return None


def measure_target_with_papi(
    target_name: str,
    device: torch.device,
    n: int,
    event_names: list[str],
    warmup: int,
    repeats: int,
    inner_iters: int,
    cache_line_bytes: int,
    dtype_bytes: int,
    estimate_missing: bool,
) -> dict:
    papi_high, resolved_event_names, event_codes = load_papi_events(event_names)
    fn = br.build_target(target_name, device=device, n=n)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)

        rows = []
        for _ in range(repeats):
            papi_high.start_counters(event_codes)
            start = time.perf_counter()
            for _ in range(inner_iters):
                fn()
            synchronize(device)
            elapsed_s = time.perf_counter() - start
            values = papi_high.stop_counters()
            counters = dict(zip(resolved_event_names, values))
            rows.append((elapsed_s, counters))

    elapsed_per_call = [elapsed / inner_iters for elapsed, _ in rows]
    counter_lists = {
        name: [counters[name] for _, counters in rows]
        for name in resolved_event_names
    }
    counter_means = {
        name: statistics.fmean(values)
        for name, values in counter_lists.items()
    }

    flops_per_call = measured_flops(counter_means, inner_iters)
    bytes_per_call = measured_memory_bytes(counter_means, inner_iters, cache_line_bytes)
    cost_source = "papi"

    estimate = estimate_target_cost(target_name, n, dtype_bytes=dtype_bytes)
    if estimate_missing and estimate is not None:
        if flops_per_call is None:
            flops_per_call = float(estimate["estimated_flops_per_call"])
            cost_source = "estimated_missing_flops"
        if bytes_per_call is None:
            bytes_per_call = float(estimate["estimated_bytes_per_call"])
            cost_source = "estimated_missing_bytes"

    mean_s = statistics.fmean(elapsed_per_call)
    row = {
        "target": target_name,
        "n": n,
        "device": str(device),
        "warmup": warmup,
        "repeats": repeats,
        "inner_iters": inner_iters,
        "mean_s": mean_s,
        "p50_s": statistics.median(elapsed_per_call),
        "min_s": min(elapsed_per_call),
        "max_s": max(elapsed_per_call),
        "papi_events": resolved_event_names,
        "papi_counters_mean_total_region": counter_means,
        "flops_per_call": flops_per_call,
        "memory_bytes_per_call": bytes_per_call,
        "cost_source": cost_source,
    }
    if flops_per_call is not None:
        row["gflops"] = flops_per_call / mean_s / 1e9
    if flops_per_call is not None and bytes_per_call is not None and bytes_per_call > 0:
        row["arithmetic_intensity"] = flops_per_call / bytes_per_call
    if estimate is not None:
        row.update(estimate)
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure benchmark_repo targets with PAPI counters for roofline plots."
    )
    parser.add_argument(
        "--target",
        choices=["all", *br.get_target_names()],
        action="append",
        default=None,
        help="Repeat for multiple targets, or 'all'. Default: all.",
    )
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--event", action="append", default=None)
    parser.add_argument("--cache-line-bytes", type=int, default=64)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--estimate-missing", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    if device.type != "cpu":
        raise SystemExit("PAPI roofline measurements are intended for CPU targets.")

    event_names = args.event or DEFAULT_EVENTS
    targets = args.target or ["all"]
    target_names = br.get_target_names() if "all" in targets else targets

    results = []
    for name in target_names:
        row = measure_target_with_papi(
            target_name=name,
            device=device,
            n=args.n,
            event_names=event_names,
            warmup=args.warmup,
            repeats=args.repeats,
            inner_iters=args.inner_iters,
            cache_line_bytes=args.cache_line_bytes,
            dtype_bytes=args.dtype_bytes,
            estimate_missing=args.estimate_missing,
        )
        results.append(row)
        ai = row.get("arithmetic_intensity")
        gf = row.get("gflops")
        print(
            f"{name:32s} {row['mean_s'] * 1e3:10.3f} ms "
            f"{gf if gf is not None else float('nan'):10.3f} GF/s "
            f"{ai if ai is not None else float('nan'):10.3f} FLOP/B"
        )

    payload = {
        "n": args.n,
        "cfg": br.make_cfg(args.n),
        "device": str(device),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "inner_iters": args.inner_iters,
        "events": event_names,
        "cache_line_bytes": args.cache_line_bytes,
        "dtype_bytes": args.dtype_bytes,
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
