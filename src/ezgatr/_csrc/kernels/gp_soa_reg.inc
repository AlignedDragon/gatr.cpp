// Derived from gp_soa_avx.inc: same FMA structure, __m256 register operands
// instead of memory (load_ps(xb[j]) -> xb[j], store_ps(ob[i],v) -> ob[i]=v).
#if defined(__AVX2__) && defined(__FMA__)
static inline __attribute__((always_inline)) void gp_soa_block_f32_reg(
        const __m256 xb[16], const __m256 yb[16], __m256 ob[16]) {
    {  // blade 00
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[0], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[2], a1);
        a0 = _mm256_fmadd_ps(xb[3], yb[3], a0);
        a1 = _mm256_fmadd_ps(xb[4], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[8], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[9], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[10], a0);
        a1 = _mm256_fnmadd_ps(xb[14], yb[14], a1);
        ob[0] = _mm256_add_ps(a0, a1);
    }
    {  // blade 01
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[1], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[5], a0);
        a1 = _mm256_fnmadd_ps(xb[3], yb[6], a1);
        a0 = _mm256_fnmadd_ps(xb[4], yb[7], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[2], a1);
        a0 = _mm256_fmadd_ps(xb[6], yb[3], a0);
        a1 = _mm256_fmadd_ps(xb[7], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[11], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[12], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[13], a0);
        a1 = _mm256_fnmadd_ps(xb[11], yb[8], a1);
        a0 = _mm256_fnmadd_ps(xb[12], yb[9], a0);
        a1 = _mm256_fnmadd_ps(xb[13], yb[10], a1);
        a0 = _mm256_fmadd_ps(xb[14], yb[15], a0);
        a1 = _mm256_fnmadd_ps(xb[15], yb[14], a1);
        ob[1] = _mm256_add_ps(a0, a1);
    }
    {  // blade 02
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[2], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[3], yb[8], a0);
        a1 = _mm256_fnmadd_ps(xb[4], yb[9], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[3], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[14], a0);
        a1 = _mm256_fnmadd_ps(xb[14], yb[10], a1);
        ob[2] = _mm256_add_ps(a0, a1);
    }
    {  // blade 03
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[3], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[3], yb[0], a0);
        a1 = _mm256_fnmadd_ps(xb[4], yb[10], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[2], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[14], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[4], a0);
        a1 = _mm256_fmadd_ps(xb[14], yb[9], a1);
        ob[3] = _mm256_add_ps(a0, a1);
    }
    {  // blade 04
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[4], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[9], a1);
        a0 = _mm256_fmadd_ps(xb[3], yb[10], a0);
        a1 = _mm256_fmadd_ps(xb[4], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[14], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[2], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[3], a0);
        a1 = _mm256_fnmadd_ps(xb[14], yb[8], a1);
        ob[4] = _mm256_add_ps(a0, a1);
    }
    {  // blade 05
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[5], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[2], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[1], a0);
        a1 = _mm256_fmadd_ps(xb[3], yb[11], a1);
        a0 = _mm256_fmadd_ps(xb[4], yb[12], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[6], yb[8], a0);
        a1 = _mm256_fnmadd_ps(xb[7], yb[9], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[6], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[7], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[15], a0);
        a1 = _mm256_fmadd_ps(xb[11], yb[3], a1);
        a0 = _mm256_fmadd_ps(xb[12], yb[4], a0);
        a1 = _mm256_fnmadd_ps(xb[13], yb[14], a1);
        a0 = _mm256_fmadd_ps(xb[14], yb[13], a0);
        a1 = _mm256_fnmadd_ps(xb[15], yb[10], a1);
        ob[5] = _mm256_add_ps(a0, a1);
    }
    {  // blade 06
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[6], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[3], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[11], a0);
        a1 = _mm256_fnmadd_ps(xb[3], yb[1], a1);
        a0 = _mm256_fmadd_ps(xb[4], yb[13], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[6], yb[0], a0);
        a1 = _mm256_fnmadd_ps(xb[7], yb[10], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[5], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[15], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[7], a0);
        a1 = _mm256_fnmadd_ps(xb[11], yb[2], a1);
        a0 = _mm256_fmadd_ps(xb[12], yb[14], a0);
        a1 = _mm256_fmadd_ps(xb[13], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[14], yb[12], a0);
        a1 = _mm256_fmadd_ps(xb[15], yb[9], a1);
        ob[6] = _mm256_add_ps(a0, a1);
    }
    {  // blade 07
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[7], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[12], a0);
        a1 = _mm256_fnmadd_ps(xb[3], yb[13], a1);
        a0 = _mm256_fnmadd_ps(xb[4], yb[1], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[9], a1);
        a0 = _mm256_fmadd_ps(xb[6], yb[10], a0);
        a1 = _mm256_fmadd_ps(xb[7], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[15], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[5], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[6], a0);
        a1 = _mm256_fnmadd_ps(xb[11], yb[14], a1);
        a0 = _mm256_fnmadd_ps(xb[12], yb[2], a0);
        a1 = _mm256_fnmadd_ps(xb[13], yb[3], a1);
        a0 = _mm256_fmadd_ps(xb[14], yb[11], a0);
        a1 = _mm256_fnmadd_ps(xb[15], yb[8], a1);
        ob[7] = _mm256_add_ps(a0, a1);
    }
    {  // blade 08
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[8], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[3], a1);
        a0 = _mm256_fnmadd_ps(xb[3], yb[2], a0);
        a1 = _mm256_fmadd_ps(xb[4], yb[14], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[0], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[10], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[9], a0);
        a1 = _mm256_fmadd_ps(xb[14], yb[4], a1);
        ob[8] = _mm256_add_ps(a0, a1);
    }
    {  // blade 09
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[9], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[3], yb[14], a0);
        a1 = _mm256_fnmadd_ps(xb[4], yb[2], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[10], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[8], a0);
        a1 = _mm256_fnmadd_ps(xb[14], yb[3], a1);
        ob[9] = _mm256_add_ps(a0, a1);
    }
    {  // blade 10
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[10], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[14], a1);
        a0 = _mm256_fmadd_ps(xb[3], yb[4], a0);
        a1 = _mm256_fnmadd_ps(xb[4], yb[3], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[9], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[0], a0);
        a1 = _mm256_fmadd_ps(xb[14], yb[2], a1);
        ob[10] = _mm256_add_ps(a0, a1);
    }
    {  // blade 11
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[11], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[8], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[6], a0);
        a1 = _mm256_fmadd_ps(xb[3], yb[5], a1);
        a0 = _mm256_fnmadd_ps(xb[4], yb[15], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[3], a1);
        a0 = _mm256_fnmadd_ps(xb[6], yb[2], a0);
        a1 = _mm256_fmadd_ps(xb[7], yb[14], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[1], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[13], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[12], a0);
        a1 = _mm256_fmadd_ps(xb[11], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[12], yb[10], a0);
        a1 = _mm256_fmadd_ps(xb[13], yb[9], a1);
        a0 = _mm256_fnmadd_ps(xb[14], yb[7], a0);
        a1 = _mm256_fmadd_ps(xb[15], yb[4], a1);
        ob[11] = _mm256_add_ps(a0, a1);
    }
    {  // blade 12
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[12], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[9], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[7], a0);
        a1 = _mm256_fmadd_ps(xb[3], yb[15], a1);
        a0 = _mm256_fmadd_ps(xb[4], yb[5], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[6], yb[14], a0);
        a1 = _mm256_fnmadd_ps(xb[7], yb[2], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[13], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[1], a1);
        a0 = _mm256_fnmadd_ps(xb[10], yb[11], a0);
        a1 = _mm256_fmadd_ps(xb[11], yb[10], a1);
        a0 = _mm256_fmadd_ps(xb[12], yb[0], a0);
        a1 = _mm256_fnmadd_ps(xb[13], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[14], yb[6], a0);
        a1 = _mm256_fnmadd_ps(xb[15], yb[3], a1);
        ob[12] = _mm256_add_ps(a0, a1);
    }
    {  // blade 13
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[13], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[10], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[15], a0);
        a1 = _mm256_fnmadd_ps(xb[3], yb[7], a1);
        a0 = _mm256_fmadd_ps(xb[4], yb[6], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[14], a1);
        a0 = _mm256_fmadd_ps(xb[6], yb[4], a0);
        a1 = _mm256_fnmadd_ps(xb[7], yb[3], a1);
        a0 = _mm256_fnmadd_ps(xb[8], yb[12], a0);
        a1 = _mm256_fmadd_ps(xb[9], yb[11], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[1], a0);
        a1 = _mm256_fnmadd_ps(xb[11], yb[9], a1);
        a0 = _mm256_fmadd_ps(xb[12], yb[8], a0);
        a1 = _mm256_fmadd_ps(xb[13], yb[0], a1);
        a0 = _mm256_fnmadd_ps(xb[14], yb[5], a0);
        a1 = _mm256_fmadd_ps(xb[15], yb[2], a1);
        ob[13] = _mm256_add_ps(a0, a1);
    }
    {  // blade 14
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[14], a0);
        a1 = _mm256_fmadd_ps(xb[2], yb[10], a1);
        a0 = _mm256_fnmadd_ps(xb[3], yb[9], a0);
        a1 = _mm256_fmadd_ps(xb[4], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[4], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[3], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[2], a0);
        a1 = _mm256_fmadd_ps(xb[14], yb[0], a1);
        ob[14] = _mm256_add_ps(a0, a1);
    }
    {  // blade 15
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        a0 = _mm256_fmadd_ps(xb[0], yb[15], a0);
        a1 = _mm256_fmadd_ps(xb[1], yb[14], a1);
        a0 = _mm256_fnmadd_ps(xb[2], yb[13], a0);
        a1 = _mm256_fmadd_ps(xb[3], yb[12], a1);
        a0 = _mm256_fnmadd_ps(xb[4], yb[11], a0);
        a1 = _mm256_fmadd_ps(xb[5], yb[10], a1);
        a0 = _mm256_fnmadd_ps(xb[6], yb[9], a0);
        a1 = _mm256_fmadd_ps(xb[7], yb[8], a1);
        a0 = _mm256_fmadd_ps(xb[8], yb[7], a0);
        a1 = _mm256_fnmadd_ps(xb[9], yb[6], a1);
        a0 = _mm256_fmadd_ps(xb[10], yb[5], a0);
        a1 = _mm256_fmadd_ps(xb[11], yb[4], a1);
        a0 = _mm256_fnmadd_ps(xb[12], yb[3], a0);
        a1 = _mm256_fmadd_ps(xb[13], yb[2], a1);
        a0 = _mm256_fnmadd_ps(xb[14], yb[1], a0);
        a1 = _mm256_fmadd_ps(xb[15], yb[0], a1);
        ob[15] = _mm256_add_ps(a0, a1);
    }
}
#endif  // __AVX2__ && __FMA__
