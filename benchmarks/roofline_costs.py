from __future__ import annotations


def cfg_from_n(n: int, formula: str = "old") -> dict[str, int]:
    if formula == "new":
        return {"batch": 2 * n, "tokens": 128 * n, "channels": 8, "heads": 4 * n}
    # legacy formula used in version_sweep_1thread.json / run_project_version_sweep.py
    return {"batch": 64, "tokens": 16 * n, "channels": 4 * n, "heads": 128 * n}


def estimate_target_cost(target: str, n: int, dtype_bytes: int = 4, formula: str = "old") -> dict[str, float | str] | None:
    """Return rough per-call FLOP/byte estimates for roofline fallback plots.

    These are intentionally simple analytical estimates. Prefer measured PAPI
    counters when available; use these when a platform does not expose a useful
    floating-point or memory event.
    """

    cfg = cfg_from_n(n, formula=formula)
    batch = cfg["batch"]
    tokens = cfg["tokens"]
    channels = cfg["channels"]
    heads = cfg["heads"]
    mv_count = batch * tokens * channels

    if target.startswith("geometric_product"):
        dense = target in {"geometric_product", "geometric_product_v0"}
        flops_per_mv = 8192 if dense else 384
        bytes_per_mv = 3 * 16 * dtype_bytes
        return {
            "estimated_flops_per_call": float(mv_count * flops_per_mv),
            "estimated_bytes_per_call": float(mv_count * bytes_per_mv),
            "cost_model": "geometric product nonzero table estimate",
        }

    if target.startswith("equi_join"):
        dense = target in {"equi_join", "equi_join_v0"}
        flops_per_mv = 8192 if dense else 384
        bytes_per_mv = 4 * 16 * dtype_bytes
        return {
            "estimated_flops_per_call": float(mv_count * flops_per_mv),
            "estimated_bytes_per_call": float(mv_count * bytes_per_mv),
            "cost_model": "join nonzero table estimate",
        }

    if target.startswith("equi_linear"):
        # For each output blade, combine roughly 9 basis terms across all input
        # channels. Count multiply-add as 2 FLOPs.
        flops = batch * tokens * channels * 16 * channels * 9 * 2
        bytes_ = batch * tokens * (2 * channels * 16 + channels) * dtype_bytes
        bytes_ += channels * channels * 9 * dtype_bytes
        return {
            "estimated_flops_per_call": float(flops),
            "estimated_bytes_per_call": float(bytes_),
            "cost_model": "linear channel-mixing estimate",
        }

    if target.startswith("equi_rms_norm"):
        # Approximate two passes over each multivector: sum of squares, then
        # normalization/weighting. Div/sqrt cost is not modeled separately here.
        flops_per_mv = 16 * 2 + 16 * 2
        bytes_per_mv = 2 * 16 * dtype_bytes
        return {
            "estimated_flops_per_call": float(mv_count * flops_per_mv),
            "estimated_bytes_per_call": float(mv_count * bytes_per_mv),
            "cost_model": "RMSNorm elementwise/reduction estimate",
        }

    if target.startswith("scaler_gated_gelu"):
        flops_per_mv = 16 * 8
        bytes_per_mv = 2 * 16 * dtype_bytes
        return {
            "estimated_flops_per_call": float(mv_count * flops_per_mv),
            "estimated_bytes_per_call": float(mv_count * bytes_per_mv),
            "cost_model": "GELU polynomial elementwise estimate, nonlinear cost omitted",
        }

    if target.startswith("equi_geometric_attention") or target == "equi_geometric_attention_mv_only":
        # QK width is 12C, value width is 16C. Dense attention dominates:
        # QK^T: 2 * B * H * T^2 * 12C
        # AV:   2 * B * H * T^2 * 16C
        dense_flops = 56 * batch * heads * tokens * tokens * channels
        qk_setup_flops = 496 * batch * heads * tokens * channels
        qkv_bytes = 3 * batch * heads * tokens * channels * 16 * dtype_bytes
        compact_qk_bytes = 2 * batch * heads * tokens * channels * 12 * dtype_bytes
        return {
            "estimated_flops_per_call": float(dense_flops + qk_setup_flops),
            "estimated_bytes_per_call": float(qkv_bytes + compact_qk_bytes),
            "cost_model": "attention dense QK/AV plus QK setup estimate",
        }

    return None
