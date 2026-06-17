"""Correctness + benchmark for the v4 AVX-512 hand-SIMD GP / equi_join kernels.

v4 is the SoA kernel with a *native* 512-bit 16x16 transpose (one multivector =
one __m512 of 16 fp32). It exists to beat the AVX-512 *autovectorized* v2 on the
Tiger Lake target, where the older hand-SIMD v3 (AVX2, 256-bit) loses because it
leaves half the SIMD width on the floor. On a non-AVX-512 host v4 transparently
falls back to the AVX2 v3 kernel, so this script runs (and the correctness check
passes) anywhere — but the speedup it is meant to demonstrate only shows up on an
AVX-512 build (e.g. `-march=native` on the i7-1165G7).

Run on the Tiger Lake box, single thread:

    python scripts/build_ext.py            # build with -march=native (=> AVX-512 on TGL)
    python benchmarks/bench_v4.py

Reports, per op and per size: v2 (autovec), v3 (AVX2 hand), v4 (AVX-512 hand) in
ms, plus the v4/v2 and v4/v3 speedups. The headline number is v4-over-v2: > 1.0x
means hand-written AVX-512 finally beats the autovectorizer for GP / join.
"""
import argparse
import time

import torch

import ezgatr.opt as o

torch.set_num_threads(1)


def cpu_has_avx512() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            return "avx512f" in f.read()
    except OSError:
        return False


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
    # include sizes that exercise the 16-MV main loop and the scalar remainder.
    for n in (16, 17, 31, 2048, 8193):
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
    avx512 = cpu_has_avx512()
    print(f"== benchmark (single thread; AVX-512 active = {avx512}) ==")
    if not avx512:
        print("  NOTE: no AVX-512 on this host -> v4 runs the AVX2 v3 fallback, so v4≈v3.")
        print("        Run this on the Tiger Lake i7-1165G7 to see the AVX-512 win.\n")
    hdr = f"{'op':18s}{'N':>9s}{'v2(auto)':>11s}{'v3(AVX2)':>11s}{'v4(512)':>11s}{'v4/v2':>9s}{'v4/v3':>9s}"
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
            print(f"{name:18s}{n:>9d}{t2:>11.4f}{t3:>11.4f}{t4:>11.4f}"
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
