from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_repo as br  # noqa: E402


BASELINE = {"batch": 4, "tokens": 128, "channels": 16, "heads": 4}

SWEEPS = {
    "batch":    [1, 2, 4, 8, 16],
    "tokens":   [32, 64, 128, 256, 512],
    "channels": [8, 16, 32, 64],
    "heads":    [1, 2, 4, 8],
}


def measure_with_config(
    target_name: str,
    device: torch.device,
    cfg: dict,
    warmup: int,
    repeats: int,
    inner_iters: int,
) -> dict:
    key = "_sweep_tmp"
    br.PRESETS[key] = cfg
    try:
        row = br.measure_target(
            target_name=target_name,
            device=device,
            preset=key,
            warmup=warmup,
            repeats=repeats,
            inner_iters=inner_iters,
        )
    finally:
        br.PRESETS.pop(key, None)

    row.pop("preset", None)
    row["batch"] = cfg["batch"]
    row["tokens"] = cfg["tokens"]
    row["channels"] = cfg["channels"]
    row["heads"] = cfg["heads"]
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-at-a-time (OAT) sweep runner. Holds a baseline config fixed "
            "and varies one axis at a time across SWEEPS[axis]."
        )
    )
    parser.add_argument(
        "--target",
        choices=["all", *br.get_target_names()],
        action="append",
        default=None,
        help="Repeat for multiple targets, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--sweep",
        choices=sorted(SWEEPS),
        action="append",
        default=None,
        help="Axis to sweep. Repeatable. Default: every axis.",
    )
    parser.add_argument("--batch", type=int, help="Override baseline batch.")
    parser.add_argument("--tokens", type=int, help="Override baseline tokens.")
    parser.add_argument("--channels", type=int, help="Override baseline channels.")
    parser.add_argument("--heads", type=int, help="Override baseline heads.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inner-iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def print_sweep_table(axis: str, rows: list[dict]) -> None:
    print(f"\n=== sweep axis: {axis} ===")
    print(
        f"{'target':28} {axis:>8} "
        f"{'mean_ms':>10} {'p50_ms':>10} "
        f"{'min_ms':>10} {'max_ms':>10}"
    )
    for row in rows:
        print(
            f"{row['target']:28} {row[axis]:>8} "
            f"{row['mean_ms']:10.3f} "
            f"{row['p50_ms']:10.3f} "
            f"{row['min_ms']:10.3f} "
            f"{row['max_ms']:10.3f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    baseline = dict(BASELINE)
    for k in ("batch", "tokens", "channels", "heads"):
        v = getattr(args, k)
        if v is not None:
            baseline[k] = v

    targets = args.target or ["all"]
    target_names = br.get_target_names() if "all" in targets else targets

    axes = args.sweep or list(SWEEPS.keys())

    all_results: dict[str, list[dict]] = {}
    for axis in axes:
        rows: list[dict] = []
        for value in SWEEPS[axis]:
            cfg = dict(baseline)
            cfg[axis] = value
            for name in target_names:
                row = measure_with_config(
                    target_name=name,
                    device=device,
                    cfg=cfg,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    inner_iters=args.inner_iters,
                )
                row["sweep_axis"] = axis
                rows.append(row)
        all_results[axis] = rows
        print_sweep_table(axis, rows)

    print(
        f"\nbaseline={baseline} device={device} "
        f"threads={args.threads} inner_iters={args.inner_iters}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": str(device),
            "threads": args.threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "inner_iters": args.inner_iters,
            "baseline": baseline,
            "sweeps": {axis: SWEEPS[axis] for axis in axes},
            "results": all_results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
