from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel
from ezgatr.nn.functional import (
    geometric_product as geometric_product_py,
    equi_join as equi_join_py,
    outer_product as outer_product_py,
    inner_product as inner_product_py,
    equi_linear as equi_linear_py,
    equi_rms_norm as equi_rms_norm_py,
    scaler_gated_gelu as scaler_gated_gelu_py,
    equi_geometric_attention as equi_geometric_attention_py,
    equi_geometric_attention_cpp,
    equi_geometric_attention_cpp_ver_0,
    equi_geometric_attention_cpp_ver_1,
    equi_geometric_attention_cpp_ver_2,
)

from ezgatr.opt import (
    geometric_product_ver_0,
    equi_join_ver_0,
    outer_product_ver_0,
    inner_product_ver_0,
    equi_linear_ver_0,
    equi_rms_norm_ver_0,
    scaler_gated_gelu_ver_0,
    equi_geometric_attention_ver_0,
    geometric_product_ver_1,
    equi_join_ver_1,
    outer_product_ver_1,
    inner_product_ver_1,
    equi_linear_ver_1,
    equi_rms_norm_ver_1,
    scaler_gated_gelu_ver_1,
    equi_geometric_attention_ver_1,
    geometric_product_ver_2,
    equi_join_ver_2,
    outer_product_ver_2,
    inner_product_ver_2,
    equi_linear_ver_2,
    equi_rms_norm_ver_2,
    scaler_gated_gelu_ver_2,
    equi_geometric_attention_ver_2,
    geometric_product_ver_3,
    equi_join_ver_3,
    outer_product_ver_3,
    inner_product_ver_3,
    equi_linear_ver_3,
    equi_rms_norm_ver_3,
    scaler_gated_gelu_ver_3,
)


PRESETS = {
    "tiny": {"batch": 2, "tokens": 32, "channels": 8, "heads": 2},
    "small": {"batch": 4, "tokens": 128, "channels": 16, "heads": 4},
    "medium": {"batch": 4, "tokens": 256, "channels": 32, "heads": 4},
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(device: torch.device, preset: str) -> dict[str, torch.Tensor]:
    cfg = PRESETS[preset]
    batch = cfg["batch"]
    tokens = cfg["tokens"]
    channels = cfg["channels"]
    heads = cfg["heads"]

    return {
        "mv": torch.randn(batch, tokens, channels, 16, device=device),
        "mv2": torch.randn(batch, tokens, channels, 16, device=device),
        "attn_q": torch.randn(batch, heads, tokens, channels, 16, device=device),
        "attn_k": torch.randn(batch, heads, tokens, channels, 16, device=device),
        "attn_v": torch.randn(batch, heads, tokens, channels, 16, device=device),
        "lin_w": torch.randn(channels, channels, 9, device=device),
        "lin_b": torch.randn(channels, device=device),
        "norm_w": torch.ones(channels, device=device),
        "model_in": torch.randn(batch, tokens, 1, 16, device=device),
    }


def build_model(device: torch.device, preset: str) -> MVOnlyGATrModel:
    cfg = PRESETS[preset]
    model_cfg = MVOnlyGATrConfig(
        num_layers=2,
        size_context=cfg["tokens"],
        size_channels_in=1,
        size_channels_out=1,
        size_channels_hidden=max(8, cfg["channels"]),
        size_channels_intermediate=max(8, cfg["channels"]),
        attn_num_heads=cfg["heads"],
        attn_is_causal=False,
    )
    return MVOnlyGATrModel(model_cfg).to(device).eval()


def build_target(name: str, device: torch.device, preset: str):
    inputs = make_inputs(device, preset)
    model = None

    if name == "geometric_product":
        return lambda: geometric_product_py(inputs["mv"], inputs["mv2"])
    if name == "equi_join":
        return lambda: equi_join_py(inputs["mv"], inputs["mv2"], None)
    if name == "outer_product":
        return lambda: outer_product_py(inputs["mv"], inputs["mv2"])
    if name == "inner_product":
        return lambda: inner_product_py(inputs["mv"], inputs["mv2"])
    if name == "equi_linear":
        return lambda: equi_linear_py(inputs["mv"], inputs["lin_w"], inputs["lin_b"])
    if name == "equi_rms_norm":
        return lambda: equi_rms_norm_py(inputs["mv"], inputs["norm_w"])
    if name == "scaler_gated_gelu":
        return lambda: scaler_gated_gelu_py(inputs["mv"], "tanh")
    if name == "equi_geometric_attention":
        return lambda: equi_geometric_attention_py(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )
    if name == "equi_geometric_attention_cpp":
        return lambda: equi_geometric_attention_cpp(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )
    if name == "equi_geometric_attention_cpp_ver_0":
        return lambda: equi_geometric_attention_cpp_ver_0(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )
    if name == "equi_geometric_attention_cpp_ver_1":
        return lambda: equi_geometric_attention_cpp_ver_1(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )
    if name == "equi_geometric_attention_cpp_ver_2":
        return lambda: equi_geometric_attention_cpp_ver_2(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )
    if name == "mv_only_gatr_model":
        model = build_model(device, preset)
        return lambda: model(inputs["model_in"])

    if name == "geometric_product_ver_0":
        return lambda: geometric_product_ver_0(inputs["mv"], inputs["mv2"])
    if name == "equi_join_ver_0":
        return lambda: equi_join_ver_0(inputs["mv"], inputs["mv2"], None)
    if name == "outer_product_ver_0":
        return lambda: outer_product_ver_0(inputs["mv"], inputs["mv2"])
    if name == "inner_product_ver_0":
        return lambda: inner_product_ver_0(inputs["mv"], inputs["mv2"])
    if name == "equi_linear_ver_0":
        return lambda: equi_linear_ver_0(inputs["mv"], inputs["lin_w"], inputs["lin_b"])
    if name == "equi_rms_norm_ver_0":
        return lambda: equi_rms_norm_ver_0(inputs["mv"], inputs["norm_w"], 1e-7)
    if name == "scaler_gated_gelu_ver_0":
        return lambda: scaler_gated_gelu_ver_0(inputs["mv"], "tanh")
    if name == "equi_geometric_attention_ver_0":
        return lambda: equi_geometric_attention_ver_0(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )

    if name == "geometric_product_ver_1":
        return lambda: geometric_product_ver_1(inputs["mv"], inputs["mv2"])
    if name == "equi_join_ver_1":
        return lambda: equi_join_ver_1(inputs["mv"], inputs["mv2"], None)
    if name == "outer_product_ver_1":
        return lambda: outer_product_ver_1(inputs["mv"], inputs["mv2"])
    if name == "inner_product_ver_1":
        return lambda: inner_product_ver_1(inputs["mv"], inputs["mv2"])
    if name == "equi_linear_ver_1":
        return lambda: equi_linear_ver_1(inputs["mv"], inputs["lin_w"], inputs["lin_b"])
    if name == "equi_rms_norm_ver_1":
        return lambda: equi_rms_norm_ver_1(inputs["mv"], inputs["norm_w"], 1e-7)
    if name == "scaler_gated_gelu_ver_1":
        return lambda: scaler_gated_gelu_ver_1(inputs["mv"], "tanh")
    if name == "equi_geometric_attention_ver_1":
        return lambda: equi_geometric_attention_ver_1(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )

    if name == "geometric_product_ver_2":
        return lambda: geometric_product_ver_2(inputs["mv"], inputs["mv2"])
    if name == "equi_join_ver_2":
        return lambda: equi_join_ver_2(inputs["mv"], inputs["mv2"], None)
    if name == "outer_product_ver_2":
        return lambda: outer_product_ver_2(inputs["mv"], inputs["mv2"])
    if name == "inner_product_ver_2":
        return lambda: inner_product_ver_2(inputs["mv"], inputs["mv2"])
    if name == "equi_linear_ver_2":
        return lambda: equi_linear_ver_2(inputs["mv"], inputs["lin_w"], inputs["lin_b"])
    if name == "equi_rms_norm_ver_2":
        return lambda: equi_rms_norm_ver_2(inputs["mv"], inputs["norm_w"], 1e-7)
    if name == "scaler_gated_gelu_ver_2":
        return lambda: scaler_gated_gelu_ver_2(inputs["mv"], "tanh")
    if name == "equi_geometric_attention_ver_2":
        return lambda: equi_geometric_attention_ver_2(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )

    if name == "geometric_product_ver_3":
        return lambda: geometric_product_ver_3(inputs["mv"], inputs["mv2"])
    if name == "equi_join_ver_3":
        return lambda: equi_join_ver_3(inputs["mv"], inputs["mv2"], None)
    if name == "outer_product_ver_3":
        return lambda: outer_product_ver_3(inputs["mv"], inputs["mv2"])
    if name == "inner_product_ver_3":
        return lambda: inner_product_ver_3(inputs["mv"], inputs["mv2"])
    if name == "equi_linear_ver_3":
        return lambda: equi_linear_ver_3(inputs["mv"], inputs["lin_w"], inputs["lin_b"])
    if name == "equi_rms_norm_ver_3":
        return lambda: equi_rms_norm_ver_3(inputs["mv"], inputs["norm_w"], 1e-7)
    if name == "scaler_gated_gelu_ver_3":
        return lambda: scaler_gated_gelu_ver_3(inputs["mv"], "tanh")
    raise ValueError(f"Unknown target: {name}")


def get_target_names() -> list[str]:
    return [
        "geometric_product",
        "geometric_product_ver_0",
        "geometric_product_ver_1",
        "geometric_product_ver_2",
        "geometric_product_ver_3",
        "equi_join",
        "equi_join_ver_0",
        "equi_join_ver_1",
        "equi_join_ver_2",
        "equi_join_ver_3",
        "outer_product",
        "outer_product_ver_0",
        "outer_product_ver_1",
        "outer_product_ver_2",
        "outer_product_ver_3",
        "inner_product",
        "inner_product_ver_0",
        "inner_product_ver_1",
        "inner_product_ver_2",
        "inner_product_ver_3",
        "equi_linear",
        "equi_linear_ver_0",
        "equi_linear_ver_1",
        "equi_linear_ver_2",
        "equi_linear_ver_3",
        "equi_rms_norm",
        "equi_rms_norm_ver_0",
        "equi_rms_norm_ver_1",
        "equi_rms_norm_ver_2",
        "equi_rms_norm_ver_3",
        "equi_geometric_attention",
        "equi_geometric_attention_cpp",
        "equi_geometric_attention_cpp_ver_0",
        "equi_geometric_attention_cpp_ver_1",
        "equi_geometric_attention_cpp_ver_2",
        "equi_geometric_attention_ver_0",
        "equi_geometric_attention_ver_1",
        "equi_geometric_attention_ver_2",
        "scaler_gated_gelu",
        "scaler_gated_gelu_ver_0",
        "scaler_gated_gelu_ver_1",
        "scaler_gated_gelu_ver_2",
        "scaler_gated_gelu_ver_3",
        "mv_only_gatr_model",
    ]


def measure_target(
    target_name: str,
    device: torch.device,
    preset: str,
    warmup: int,
    repeats: int,
    inner_iters: int,
) -> dict[str, float | str | int]:
    fn = build_target(target_name, device=device, preset=preset)

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
            end = time.perf_counter()
            times_ms.append((end - start) * 1e3)

    per_iter_ms = [t / inner_iters for t in times_ms]
    return {
        "target": target_name,
        "preset": preset,
        "device": str(device),
        "warmup": warmup,
        "repeats": repeats,
        "inner_iters": inner_iters,
        "mean_ms": statistics.fmean(per_iter_ms),
        "p50_ms": statistics.median(per_iter_ms),
        "min_ms": min(per_iter_ms),
        "max_ms": max(per_iter_ms),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["all", *get_target_names()], default="all")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inner-iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def print_table(results: list[dict[str, float | str | int]]) -> None:
    print(
        f"{'target':28} {'mean_ms':>10} {'p50_ms':>10} "
        f"{'min_ms':>10} {'max_ms':>10}"
    )
    for row in results:
        print(
            f"{row['target']:28} "
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

    target_names = get_target_names() if args.target == "all" else [args.target]
    results = [
        measure_target(
            target_name=name,
            device=device,
            preset=args.preset,
            warmup=args.warmup,
            repeats=args.repeats,
            inner_iters=args.inner_iters,
        )
        for name in target_names
    ]

    print(
        f"preset={args.preset} device={device} "
        f"threads={args.threads} inner_iters={args.inner_iters}"
    )
    print_table(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preset": args.preset,
            "device": str(device),
            "threads": args.threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "inner_iters": args.inner_iters,
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
