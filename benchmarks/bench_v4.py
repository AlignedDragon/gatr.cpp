"""Correctness + benchmark for the v4 register-resident SoA GP / equi_join kernels.

v4 beats the autovectorized v2 (and the earlier hand-SIMD v3) by doing the SoA
rearrange the way the Tiger Lake autovectorizer does — 8 multivectors per 256-bit
ymm tile, transposed and computed **in registers** with no xb/yb/ob stack round-
trip — but with v3's minimal `unpck` transpose and pure-FMA block instead of the
autovec's ~190 `vpermt2ps`. It is AVX2 + FMA (the autovec never uses AVX-512 for
these ops, on Tiger Lake or anywhere), so the win shows up on any AVX2 host,
including this dev box.

    python scripts/build_ext.py            # build with -march=native
    python benchmarks/bench_v4.py

Reports, per op and per size: v2 (autovec), v3 (SoA via stack buffers), v4 (SoA in
registers) in ms, plus the v4/v2 and v4/v3 speedups (>1 = v4 is faster).
"""
import argparse
import time

import torch

import ezgatr.opt as o

torch.set_num_threads(1)


def best_ms(fn, warmup_s=0.15, measure_s=0.6, min_reps=5, max_reps=2000) -> float:
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    while time.perf_counter() - t0 < warmup_s:
        fn()
    reps = max(min_reps, min(max_reps, int(measure_s / single) if single > 0 else max_reps))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def check_correctness() -> None:
    print("== correctness (v4 vs v2 reference, fp32) ==")
    torch.manual_seed(0)
    ok = True
    # sizes that exercise the 8-MV main loop and the scalar remainder.
    for n in (8, 15, 16, 2048, 8193):
        x = torch.randn(n, 16, dtype=torch.float32)
        y = torch.randn(n, 16, dtype=torch.float32)
        ref = torch.randn(n, 16, dtype=torch.float32)
        cases = {
            "geometric_product": (o.geometric_product_v2(x, y), o.geometric_product_v4(x, y)),
            "equi_join(noref)":  (o.equi_join_v2(x, y, None), o.equi_join_v4(x, y, None)),
            "equi_join(ref)":    (o.equi_join_v2(x, y, ref), o.equi_join_v4(x, y, ref)),
        }
        for name, (a, b) in cases.items():
            md = (a - b).abs().max().item()
            tol = 1e-3 * (1.0 + a.abs().max().item())
            status = "OK" if md <= tol else "FAIL"
            ok &= md <= tol
            print(f"  N={n:<6} {name:18s} max_abs_diff={md:.3e}  [{status}]")
    print("  => ALL CORRECTNESS CHECKS PASSED\n" if ok else "  => CORRECTNESS FAILURE\n")
    assert ok, "v4 disagrees with v2 beyond fp32 tolerance"


def bench(sizes) -> None:
    print("== benchmark (single thread) ==")
    hdr = (f"{'op':18s}{'N':>9s}{'v2(auto)':>11s}{'v3(SoA-mem)':>13s}"
           f"{'v4(SoA-reg)':>13s}{'v4/v2':>9s}{'v4/v3':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for n in sizes:
        x = torch.randn(n, 16, dtype=torch.float32)
        y = torch.randn(n, 16, dtype=torch.float32)
        ref = torch.randn(n, 16, dtype=torch.float32)
        ops = {
            "geometric_product": (
                lambda: o.geometric_product_v2(x, y),
                lambda: o.geometric_product_v3(x, y),
                lambda: o.geometric_product_v4(x, y),
            ),
            "equi_join(ref)": (
                lambda: o.equi_join_v2(x, y, ref),
                lambda: o.equi_join_v3(x, y, ref),
                lambda: o.equi_join_v4(x, y, ref),
            ),
        }
        for name, (f2, f3, f4) in ops.items():
            with torch.inference_mode():
                t2, t3, t4 = best_ms(f2), best_ms(f3), best_ms(f4)
            print(f"{name:18s}{n:>9d}{t2:>11.4f}{t3:>13.4f}{t4:>13.4f}"
                  f"{t2 / t4:>8.2f}x{t3 / t4:>8.2f}x")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2048, 8192, 32768, 131072, 524288],
                    help="multivector counts to sweep (L1->DRAM at 192 B/mv).")
    ap.add_argument("--no-check", action="store_true", help="skip the correctness check")
    args = ap.parse_args()
    if not args.no_check:
        check_correctness()
    bench(args.sizes)


if __name__ == "__main__":
    main()
