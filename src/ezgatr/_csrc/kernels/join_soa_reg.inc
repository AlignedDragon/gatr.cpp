// Derived from join_soa_avx.inc: same FMA structure, __m256 register operands
// instead of memory (load_ps(xb[j]) -> xb[j], store_ps(ob[i],v) -> ob[i]=v).
#if defined(__AVX2__) && defined(__FMA__)
static inline __attribute__((always_inline)) void join_soa_block_f32_reg(
        const __m256 xb[16], const __m256 yb[16], __m256 ob[16]) {
    // Pre-load hot input blades into registers to cut repeated stack loads.
    const __m256 x11 = xb[11];
    const __m256 x12 = xb[12];
    const __m256 x13 = xb[13];
    const __m256 x14 = xb[14];
    const __m256 x15 = xb[15];
    const __m256 y11 = yb[11];
    const __m256 y12 = yb[12];
    const __m256 y13 = yb[13];
    const __m256 y14 = yb[14];
    const __m256 y15 = yb[15];
    {  // blade 00
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], y15, a0);
        a1 = _mm256_fmadd_ps(xb[1], y14, a1);
        a0 = _mm256_fnmadd_ps(xb[2], y13, a0);
        a1 = _mm256_fmadd_ps(xb[3], y12, a1);
        a0 = _mm256_fnmadd_ps(xb[4], y11, a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[10], a1);
        a0 = _mm256_fnmadd_ps(xb[6], yb[9], a0);
        a1 = _mm256_fmadd_ps(xb[7], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[7], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[6], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[5], a0);
        a1 = _mm256_fmadd_ps(x11, yb[4], a1);
        a0 = _mm256_fnmadd_ps(x12, yb[3], a0);
        a1 = _mm256_fmadd_ps(x13, yb[2], a1);
        a0 = _mm256_fnmadd_ps(x14, yb[1], a0);
        a1 = _mm256_fmadd_ps(x15, yb[0], a1);
        ob[0] = _mm256_add_ps(a0, a1);
    }
    {  // blade 01
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(xb[1], y15, a0);
        a1 = _mm256_fnmadd_ps(xb[5], y13, a1);
        a0 = _mm256_fmadd_ps(xb[6], y12, a0);
        a1 = _mm256_fnmadd_ps(xb[7], y11, a1);
        a0 = _mm256_fnmadd_ps(x11, yb[7], a0);
        a1 = _mm256_fmadd_ps(x12, yb[6], a1);
        a0 = _mm256_fnmadd_ps(x13, yb[5], a0);
        a1 = _mm256_fnmadd_ps(x15, yb[1], a1);
        ob[1] = _mm256_add_ps(a0, a1);
    }
    {  // blade 02
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(xb[2], y15, a0);
        a1 = _mm256_fnmadd_ps(xb[5], y14, a1);
        a0 = _mm256_fmadd_ps(xb[8], y12, a0);
        a1 = _mm256_fnmadd_ps(xb[9], y11, a1);
        a0 = _mm256_fnmadd_ps(x11, yb[9], a0);
        a1 = _mm256_fmadd_ps(x12, yb[8], a1);
        a0 = _mm256_fnmadd_ps(x14, yb[5], a0);
        a1 = _mm256_fnmadd_ps(x15, yb[2], a1);
        ob[2] = _mm256_add_ps(a0, a1);
    }
    {  // blade 03
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(xb[3], y15, a0);
        a1 = _mm256_fnmadd_ps(xb[6], y14, a1);
        a0 = _mm256_fmadd_ps(xb[8], y13, a0);
        a1 = _mm256_fnmadd_ps(xb[10], y11, a1);
        a0 = _mm256_fnmadd_ps(x11, yb[10], a0);
        a1 = _mm256_fmadd_ps(x13, yb[8], a1);
        a0 = _mm256_fnmadd_ps(x14, yb[6], a0);
        a1 = _mm256_fnmadd_ps(x15, yb[3], a1);
        ob[3] = _mm256_add_ps(a0, a1);
    }
    {  // blade 04
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(xb[4], y15, a0);
        a1 = _mm256_fnmadd_ps(xb[7], y14, a1);
        a0 = _mm256_fmadd_ps(xb[9], y13, a0);
        a1 = _mm256_fnmadd_ps(xb[10], y12, a1);
        a0 = _mm256_fnmadd_ps(x12, yb[10], a0);
        a1 = _mm256_fmadd_ps(x13, yb[9], a1);
        a0 = _mm256_fnmadd_ps(x14, yb[7], a0);
        a1 = _mm256_fnmadd_ps(x15, yb[4], a1);
        ob[4] = _mm256_add_ps(a0, a1);
    }
    {  // blade 05
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[5], y15, a0);
        a1 = _mm256_fmadd_ps(x11, y12, a1);
        a0 = _mm256_fnmadd_ps(x12, y11, a0);
        a1 = _mm256_fmadd_ps(x15, yb[5], a1);
        ob[5] = _mm256_add_ps(a0, a1);
    }
    {  // blade 06
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[6], y15, a0);
        a1 = _mm256_fmadd_ps(x11, y13, a1);
        a0 = _mm256_fnmadd_ps(x13, y11, a0);
        a1 = _mm256_fmadd_ps(x15, yb[6], a1);
        ob[6] = _mm256_add_ps(a0, a1);
    }
    {  // blade 07
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[7], y15, a0);
        a1 = _mm256_fmadd_ps(x12, y13, a1);
        a0 = _mm256_fnmadd_ps(x13, y12, a0);
        a1 = _mm256_fmadd_ps(x15, yb[7], a1);
        ob[7] = _mm256_add_ps(a0, a1);
    }
    {  // blade 08
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[8], y15, a0);
        a1 = _mm256_fmadd_ps(x11, y14, a1);
        a0 = _mm256_fnmadd_ps(x14, y11, a0);
        a1 = _mm256_fmadd_ps(x15, yb[8], a1);
        ob[8] = _mm256_add_ps(a0, a1);
    }
    {  // blade 09
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[9], y15, a0);
        a1 = _mm256_fmadd_ps(x12, y14, a1);
        a0 = _mm256_fnmadd_ps(x14, y12, a0);
        a1 = _mm256_fmadd_ps(x15, yb[9], a1);
        ob[9] = _mm256_add_ps(a0, a1);
    }
    {  // blade 10
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[10], y15, a0);
        a1 = _mm256_fmadd_ps(x13, y14, a1);
        a0 = _mm256_fnmadd_ps(x14, y13, a0);
        a1 = _mm256_fmadd_ps(x15, yb[10], a1);
        ob[10] = _mm256_add_ps(a0, a1);
    }
    {  // blade 11
        __m256 a0 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(x11, y15, a0);
        a0 = _mm256_fnmadd_ps(x15, y11, a0);
        ob[11] = a0;
    }
    {  // blade 12
        __m256 a0 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(x12, y15, a0);
        a0 = _mm256_fnmadd_ps(x15, y12, a0);
        ob[12] = a0;
    }
    {  // blade 13
        __m256 a0 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(x13, y15, a0);
        a0 = _mm256_fnmadd_ps(x15, y13, a0);
        ob[13] = a0;
    }
    {  // blade 14
        __m256 a0 = _mm256_setzero_ps();
        a0 = _mm256_fnmadd_ps(x14, y15, a0);
        a0 = _mm256_fnmadd_ps(x15, y14, a0);
        ob[14] = a0;
    }
    {  // blade 15
        ob[15] = _mm256_mul_ps(x15, y15);
    }
}
#endif  // __AVX2__ && __FMA__
