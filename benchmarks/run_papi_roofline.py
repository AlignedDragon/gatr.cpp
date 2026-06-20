from __future__ import annotations

import argparse
import ctypes
import json
import math
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


class SystemPapi:
    PAPI_VER_CURRENT = 0x07020000
    PAPI_OK = 0
    PAPI_NULL = -1

    def __init__(self, event_names: list[str]):
        self.lib = ctypes.CDLL("libpapi.so")
        version = self.lib.PAPI_library_init(self.PAPI_VER_CURRENT)
        if version < 0:
            raise SystemExit(f"PAPI_library_init failed with code {version}")
        self.event_names = event_names
        self.event_codes = [self._event_code(name) for name in event_names]
        self.event_set = self._create_event_set(self.event_codes)

    def _event_code(self, name: str) -> int:
        code = ctypes.c_int()
        ret = self.lib.PAPI_event_name_to_code(name.encode(), ctypes.byref(code))
        if ret != self.PAPI_OK:
            raise SystemExit(f"PAPI_event_name_to_code({name}) failed with code {ret}")
        return int(code.value)

    def _create_event_set(self, event_codes: list[int]) -> ctypes.c_int:
        event_set = ctypes.c_int(self.PAPI_NULL)
        ret = self.lib.PAPI_create_eventset(ctypes.byref(event_set))
        if ret != self.PAPI_OK:
            raise SystemExit(f"PAPI_create_eventset failed with code {ret}")
        for name, code in zip(self.event_names, event_codes):
            ret = self.lib.PAPI_add_event(event_set, code)
            if ret != self.PAPI_OK:
                raise SystemExit(f"PAPI_add_event({name}) failed with code {ret}")
        return event_set

    def measure(self, fn, inner_iters: int, device: torch.device) -> tuple[float, dict[str, int]]:
        ret = self.lib.PAPI_start(self.event_set)
        if ret != self.PAPI_OK:
            raise SystemExit(f"PAPI_start failed with code {ret}")
        start = time.perf_counter()
        for _ in range(inner_iters):
            fn()
        synchronize(device)
        elapsed_s = time.perf_counter() - start
        values = (ctypes.c_longlong * len(self.event_names))()
        ret = self.lib.PAPI_stop(self.event_set, values)
        if ret != self.PAPI_OK:
            raise SystemExit(f"PAPI_stop failed with code {ret}")
        return elapsed_s, dict(zip(self.event_names, [int(v) for v in values]))

    def close(self) -> None:
        self.lib.PAPI_cleanup_eventset(self.event_set)
        self.lib.PAPI_destroy_eventset(ctypes.byref(self.event_set))


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


def split_event_groups(event_names: list[str]) -> list[list[str]]:
    """Split events that cannot be counted together on common Intel/PAPI setups."""
    fp_events = set(FLOP_EVENTS)
    memory_events = set(MEMORY_EVENTS)
    has_fp = any(name in fp_events for name in event_names)
    has_memory = any(name in memory_events for name in event_names)
    if not (has_fp and has_memory):
        return [event_names]

    base = [name for name in event_names if name not in memory_events]
    memory = [name for name in event_names if name in memory_events]
    return [base, memory]


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
    inner_iters: int | None,
    cache_line_bytes: int,
    dtype_bytes: int,
    estimate_missing: bool,
    target_region_s: float = 0.1,
) -> dict:
    fn = br.build_target(target_name, device=device, n=n)

    with torch.inference_mode():
        # settle lazy init before probing
        for _ in range(3):
            fn()
        synchronize(device)

        # auto-size inner_iters so each region runs ~target_region_s
        if inner_iters is None:
            probe_iters = 5
            probe_start = time.perf_counter()
            for _ in range(probe_iters):
                fn()
            synchronize(device)
            per_call_s = (time.perf_counter() - probe_start) / probe_iters
            inner_iters = (
                max(1, min(200000, math.ceil(target_region_s / per_call_s)))
                if per_call_s > 0
                else 1
            )

        # warm up by whole regions
        for _ in range(warmup):
            for _ in range(inner_iters):
                fn()
        synchronize(device)

        rows = []
        event_groups = split_event_groups(event_names)
        for group_index, group in enumerate(event_groups):
            papi = SystemPapi(group)
            try:
                for _ in range(repeats):
                    elapsed_s, counters = papi.measure(fn, inner_iters, device)
                    if group_index == 0:
                        rows.append((elapsed_s, counters))
                    else:
                        rows[len(rows) - repeats + _][1].update(counters)
            finally:
                papi.close()

    elapsed_per_call = [elapsed / inner_iters for elapsed, _ in rows]
    resolved_event_names = list(dict.fromkeys(name for group in event_groups for name in group))
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
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=None,
        help="Calls per timed region. Default: auto-sized to ~--target-region-ms.",
    )
    parser.add_argument(
        "--target-region-ms",
        type=float,
        default=100.0,
        help="Target wall time per region when --inner-iters is auto.",
    )
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
    payload = {
        "n": args.n,
        "cfg": br.make_cfg(args.n),
        "device": str(device),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "inner_iters": args.inner_iters,
        "target_region_ms": args.target_region_ms,
        "events": event_names,
        "cache_line_bytes": args.cache_line_bytes,
        "dtype_bytes": args.dtype_bytes,
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
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
            target_region_s=args.target_region_ms / 1000.0,
        )
        results.append(row)
        ai = row.get("arithmetic_intensity")
        gf = row.get("gflops")
        print(
            f"{name:32s} {row['mean_s'] * 1e3:10.3f} ms "
            f"{gf if gf is not None else float('nan'):10.3f} GF/s "
            f"{ai if ai is not None else float('nan'):10.3f} FLOP/B"
        )
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"checkpointed {args.json_out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
