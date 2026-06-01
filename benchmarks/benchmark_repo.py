from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ezgatr.nets.mv_only_gatr import MVOnlyGATrConfig, MVOnlyGATrModel, MVOnlyGATrModelASL_ver_0, MVOnlyGATrModelASL_ver_1, MVOnlyGATrModelASL_ver_2#, MVOnlyGATrModelASL_ver_3
from ezgatr.nn.functional import (
    geometric_product as geometric_product_py,
    equi_join as equi_join_py,
    outer_product as outer_product_py,
    inner_product as inner_product_py,
    equi_linear as equi_linear_py,
    equi_rms_norm as equi_rms_norm_py,
    scaler_gated_gelu as scaler_gated_gelu_py,
    equi_geometric_attention as equi_geometric_attention_py,
    # equi_geometric_attention_cpp,
    # equi_geometric_attention_cpp_ver_0,
    # equi_geometric_attention_cpp_ver_1,
    # equi_geometric_attention_cpp_ver_2,
)

from ezgatr.opt import (
    #version?
    equi_geometric_attention_mv_only,

    #start

    # outer_product_ver_0,
    # inner_product_ver_0,
    geometric_product_v0,
    equi_join_v0,
    equi_linear_ver_0,
    equi_rms_norm_ver_0,
    scaler_gated_gelu_ver_0,
    equi_geometric_attention_ver_0,
    #equi_geometric_attention_mv_only_ver_0,



    # outer_product_ver_1,
    # inner_product_ver_1,
    geometric_product_v1,
    equi_join_v1,
    equi_linear_ver_1,
    equi_rms_norm_ver_1,
    scaler_gated_gelu_ver_1,
    equi_geometric_attention_ver_1,
    #equi_geometric_attention_mv_only_ver_1,


    # outer_product_ver_2,
    # inner_product_ver_2,
    geometric_product_v2,
    equi_join_v2,
    equi_linear_ver_2,
    equi_rms_norm_ver_2,
    scaler_gated_gelu_ver_2,
    equi_geometric_attention_ver_2,
    #equi_geometric_attention_mv_only_ver_2,


    # outer_product_ver_3,
    # inner_product_ver_3,
    #geometric_product_v3,
    #equi_join_v3,
    equi_linear_ver_3,
    equi_rms_norm_ver_3,
    # scaler_gated_gelu_ver_3,
    equi_geometric_attention_ver_3,
    #equi_geometric_attention_mv_only_ver_3,
)
from ezgatr import opt as _opt



def make_cfg(n: int) -> dict[str, int]:
    return {
        "batch": 64,
        "tokens": 16 * n,
        "channels": 4 * n,
        "heads": 128 * n,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(device: torch.device, n: int) -> dict[str, torch.Tensor]:
    cfg = make_cfg(n)
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


def build_model(device: torch.device, n: int):
    cfg = make_cfg(n)
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

def build_model_ver_0(device: torch.device, n: int) -> MVOnlyGATrModel:
    cfg = make_cfg(n)
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
    return MVOnlyGATrModelASL_ver_0(model_cfg).to(device).eval()

def build_model_ver_1(device: torch.device, n: int) -> MVOnlyGATrModel:
    cfg = make_cfg(n)
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
    return MVOnlyGATrModelASL_ver_1(model_cfg).to(device).eval()

def build_model_ver_2(device: torch.device, n: int) -> MVOnlyGATrModel:
    cfg = make_cfg(n)
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
    return MVOnlyGATrModelASL_ver_2(model_cfg).to(device).eval()

# def build_model_ver_3(device: torch.device, n: int) -> MVOnlyGATrModel:
#     cfg = make_cfg(n)
#     model_cfg = MVOnlyGATrConfig(
#         num_layers=2,
#         size_context=cfg["tokens"],
#         size_channels_in=1,
#         size_channels_out=1,
#         size_channels_hidden=max(8, cfg["channels"]),
#         size_channels_intermediate=max(8, cfg["channels"]),
#         attn_num_heads=cfg["heads"],
#         attn_is_causal=False,
#     )
#     return MVOnlyGATrModelASL_ver_3(model_cfg).to(device).eval()


def build_target(name: str, device: torch.device, n: int):
    inputs = make_inputs(device, n)
    model = None

    #version?
    if name == "equi_geometric_attention_mv_only":
        return lambda: equi_geometric_attention_mv_only(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )


    #start
    if name == "geometric_product":
        return lambda: geometric_product_py(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v0":
        return lambda: geometric_product_v0(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v1":
        return lambda: geometric_product_v1(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2":
        return lambda: geometric_product_v2(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_1":
        return lambda: _opt.geometric_product_v2_1(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_2":
        return lambda: _opt.geometric_product_v2_2(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_3":
        return lambda: _opt.geometric_product_v2_3(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_4":
        return lambda: _opt.geometric_product_v2_4(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_5":
        return lambda: _opt.geometric_product_v2_5(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_6":
        return lambda: _opt.geometric_product_v2_6(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v2_7":
        return lambda: _opt.geometric_product_v2_7(inputs["mv"], inputs["mv2"])
    if name == "geometric_product_v3":
        return lambda: _opt.geometric_product_v3(inputs["mv"], inputs["mv2"])
<<<<<<< HEAD
    if name == "geometric_product_v3_1":
        return lambda: _opt.geometric_product_v3_1(inputs["mv"], inputs["mv2"])

    if name == "equi_join":
        return lambda: equi_join_py(inputs["mv"], inputs["mv2"], None)
    if name == "equi_join_v0":
        return lambda: equi_join_v0(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v1":
        return lambda: equi_join_v1(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2":
        return lambda: _opt.equi_join_v2(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2_3":
        return lambda: _opt.equi_join_v2_3(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2_4":
        return lambda: _opt.equi_join_v2_4(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2_1":
        return lambda: _opt.equi_join_v2_1(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2_2":
        return lambda: _opt.equi_join_v2_2(inputs["mv"], inputs["mv2"], inputs["mv"])
=======
>>>>>>> ali/kernel-rename
    if name == "equi_join_v2_5":
        return lambda: _opt.equi_join_v2_5(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2_6":
        return lambda: _opt.equi_join_v2_6(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v2_7":
        return lambda: _opt.equi_join_v2_7(inputs["mv"], inputs["mv2"], inputs["mv"])
    if name == "equi_join_v3":
        return lambda: _opt.equi_join_v3(inputs["mv"], inputs["mv2"], inputs["mv"])
<<<<<<< HEAD
    if name == "equi_join_v3_1":
        return lambda: _opt.equi_join_v3_1(inputs["mv"], inputs["mv2"], inputs["mv"])

=======
>>>>>>> ali/kernel-rename
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
    # if name == "equi_geometric_attention_cpp":
    #     return lambda: equi_geometric_attention_cpp(
    #         inputs["attn_q"],
    #         inputs["attn_k"],
    #         inputs["attn_v"],
    #         kinds={"ipa": None, "daa": None},
    #         is_causal=False,
    #     )
    # if name == "equi_geometric_attention_cpp_ver_0":
    #     return lambda: equi_geometric_attention_cpp_ver_0(
    #         inputs["attn_q"],
    #         inputs["attn_k"],
    #         inputs["attn_v"],
    #         kinds={"ipa": None, "daa": None},
    #         is_causal=False,
    #     )
    # if name == "equi_geometric_attention_cpp_ver_1":
    #     return lambda: equi_geometric_attention_cpp_ver_1(
    #         inputs["attn_q"],
    #         inputs["attn_k"],
    #         inputs["attn_v"],
    #         kinds={"ipa": None, "daa": None},
    #         is_causal=False,
    #     )
    # if name == "equi_geometric_attention_cpp_ver_2":
    #     return lambda: equi_geometric_attention_cpp_ver_2(
    #         inputs["attn_q"],
    #         inputs["attn_k"],
    #         inputs["attn_v"],
    #         kinds={"ipa": None, "daa": None},
    #         is_causal=False,
    #     )
    if name == "mv_only_gatr_model":
        model = build_model(device, n)
        return lambda: model(inputs["model_in"])
    if name == "mv_only_gatr_model_ver_0":
        model = build_model_ver_0(device, n)
        return lambda: model(inputs["model_in"])
    if name == "mv_only_gatr_model_ver_1":
        model = build_model_ver_1(device, n)
        return lambda: model(inputs["model_in"])
    if name == "mv_only_gatr_model_ver_2":
        model = build_model_ver_2(device, n)
        return lambda: model(inputs["model_in"])
    if name == "mv_only_gatr_model_ver_3":
        model = build_model_ver_3(device, n)
        return lambda: model(inputs["model_in"])
    # if name == "outer_product_ver_0":
    #     return lambda: outer_product_ver_0(inputs["mv"], inputs["mv2"])
    # if name == "inner_product_ver_0":
    #     return lambda: inner_product_ver_0(inputs["mv"], inputs["mv2"])
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

    # if name == "outer_product_ver_1":
    #     return lambda: outer_product_ver_1(inputs["mv"], inputs["mv2"])
    # if name == "inner_product_ver_1":
    #     return lambda: inner_product_ver_1(inputs["mv"], inputs["mv2"])
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

    # if name == "outer_product_ver_2":
    #     return lambda: outer_product_ver_2(inputs["mv"], inputs["mv2"])
    # if name == "inner_product_ver_2":
    #     return lambda: inner_product_ver_2(inputs["mv"], inputs["mv2"])
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

    # if name == "outer_product_ver_3":
    #     return lambda: outer_product_ver_3(inputs["mv"], inputs["mv2"])
    # if name == "inner_product_ver_3":
    #     return lambda: inner_product_ver_3(inputs["mv"], inputs["mv2"])
    if name == "equi_linear_ver_3":
        return lambda: equi_linear_ver_3(inputs["mv"], inputs["lin_w"], inputs["lin_b"])
    if name == "equi_rms_norm_ver_3":
        return lambda: equi_rms_norm_ver_3(inputs["mv"], inputs["norm_w"], 1e-7)
    # if name == "scaler_gated_gelu_ver_3":
    #     return lambda: scaler_gated_gelu_ver_3(inputs["mv"], "tanh")
    if name == "equi_geometric_attention_ver_3":
        return lambda: equi_geometric_attention_ver_3(
            inputs["attn_q"],
            inputs["attn_k"],
            inputs["attn_v"],
            kinds={"ipa": None, "daa": None},
            is_causal=False,
        )
    raise ValueError(f"Unknown target: {name}")


def get_target_names() -> list[str]:
    return [
        #version?
        "equi_geometric_attention_mv_only",
        #start
        "geometric_product",
        "geometric_product_v0",
        "geometric_product_v1",
        "geometric_product_v2",
        "geometric_product_v2_3",
        "geometric_product_v2_4",
        "geometric_product_v2_1",
        "geometric_product_v2_2",
        "geometric_product_v2_5",
        "geometric_product_v2_6",
        "geometric_product_v2_7",
        "geometric_product_v3",
        "equi_join",
        "equi_join_v0",
        "equi_join_v1",
        "equi_join_v2",
        "equi_join_v2_3",
        "equi_join_v2_4",
        "equi_join_v2_1",
        "equi_join_v2_2",
        "equi_join_v2_5",
        "equi_join_v2_6",
        "equi_join_v2_7",
        "equi_join_v3",
        "outer_product",
        "inner_product",
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
        #"equi_geometric_attention_cpp",
        # "equi_geometric_attention_cpp_ver_0",
        # "equi_geometric_attention_cpp_ver_1",
        # "equi_geometric_attention_cpp_ver_2",
        # "equi_geometric_attention_ver_0",
        # "equi_geometric_attention_ver_1",
        # "equi_geometric_attention_ver_2",
        # "equi_geometric_attention_ver_3",
        "scaler_gated_gelu",
        "scaler_gated_gelu_ver_0",
        "scaler_gated_gelu_ver_1",
        "scaler_gated_gelu_ver_2",
        #"scaler_gated_gelu_ver_3",
        # "mv_only_gatr_model",
        #"mv_only_gatr_model_ver_0",
        #"mv_only_gatr_model_ver_1",
        #"mv_only_gatr_model_ver_2",
        #"mv_only_gatr_model_ver_3",
    ]


def measure_target(
    target_name: str,
    device: torch.device,
    n: int,
    warmup: int,
    repeats: int,
    inner_iters: int,
) -> dict[str, float | str | int]:
    fn = build_target(target_name, device=device, n=n)

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
        "n": n,
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
    parser.add_argument("--target", choices=["all", *get_target_names()],
                        action="append", default=None,
                        help="Repeat for multiple targets, or 'all' for everything.")
    parser.add_argument("--n",type=int,default=1)
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

    targets = args.target or ["all"]
    if "all" in targets:
        target_names = get_target_names()
    else:
        target_names = targets
    results = [
        measure_target(
            target_name=name,
            device=device,
            n=args.n,
            warmup=args.warmup,
            repeats=args.repeats,
            inner_iters=args.inner_iters,
        )
        for name in target_names
    ]

    print(
        f"n={args.n} device={device} "
        f"threads={args.threads} inner_iters={args.inner_iters}"
    )
    print_table(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": args.n,
            "cfg": make_cfg(args.n),
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






