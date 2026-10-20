#pragma once
#include "pairhmm_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace phmm {
constexpr unsigned QUAL_COUNT = 128;
constexpr unsigned MM_COUNT = QUAL_COUNT * (QUAL_COUNT + 1) / 2;
constexpr unsigned MAX_GPU_READ_LENGTH = 1024;
inline unsigned mm_index(unsigned a, unsigned b) {
    const unsigned hi = std::max(a, b), lo = std::min(a, b);
    return hi * (hi + 1) / 2 + lo;
}
inline void error_text(char *out, size_t capacity, const char *message) {
    if (out && capacity) std::snprintf(out, capacity, "%s", message);
}
inline bool valid_base(uint8_t c) {
    return c == 'A' || c == 'C' || c == 'G' || c == 'T' || c == 'N';
}
inline void validate(const phmm_read &r, const phmm_haplotype &h) {
    if (!r.length || !h.length || !r.bases || !r.quals || !r.insertion_gop ||
        !r.deletion_gop || !r.overall_gcp || !h.bases)
        throw std::invalid_argument("PairHMM requires nonempty bases and all four Phred arrays");
    for (uint32_t i = 0; i < r.length; ++i)
        if (!valid_base(r.bases[i])) throw std::invalid_argument("Read base outside A/C/G/T/N");
    for (uint32_t j = 0; j < h.length; ++j)
        if (!valid_base(h.bases[j])) throw std::invalid_argument("Haplotype base outside A/C/G/T/N");
}

/* Independently implemented from the GKL 0.8.6 Context.h mathematical contract:
 * quality &127; Jacobian table step 1e-4, tolerance 8; rounded table lookup;
 * separate float/double tables. Source URLs and blob hashes are in SOURCES.md.
 * Computing 1-p_ins-p_del directly would change the upstream transition table. */
template <typename T> struct Tables {
    T phred[QUAL_COUNT];
    T mm[MM_COUNT];
    Tables() {
        std::vector<T> jacobian(80001);
        for (int k = 0; k <= 80000; ++k)
            jacobian[k] = static_cast<T>(std::log10(1.0 + std::pow(10.0, -k * 0.0001)));
        for (unsigned q = 0; q < QUAL_COUNT; ++q) {
            if constexpr (std::is_same<T, float>::value)
                phred[q] = ::powf(10.0f, -static_cast<float>(q) / 10.0f);
            else phred[q] = std::pow(10.0, -static_cast<double>(q) / 10.0);
        }
        const double inverse_ln10 = 1.0 / std::log(10.0);
        for (unsigned hi = 0; hi < QUAL_COUNT; ++hi) {
            for (unsigned lo = 0; lo <= hi; ++lo) {
                const T small = static_cast<T>(-0.1 * hi);
                const T big = static_cast<T>(-0.1 * lo);
                const T diff = big - small;
                T log_sum = big;
                if (diff < static_cast<T>(8.0)) {
                    const T scaled = static_cast<T>(diff * static_cast<T>(10000.0));
                    const int index = static_cast<int>(scaled + static_cast<T>(0.5));
                    log_sum = big + jacobian.at(index);
                }
                const double p = std::min(1.0, std::pow(10.0, static_cast<double>(log_sum)));
                mm[mm_index(hi, lo)] = static_cast<T>(std::pow(10.0, std::log1p(-p) * inverse_ln10));
            }
        }
    }
};
inline const Tables<float> &float_tables() { static const Tables<float> t; return t; }
inline const Tables<double> &double_tables() { static const Tables<double> t; return t; }

/* Scalar row recurrence, independent of the GPU anti-diagonal implementation.
 * Row zero deletion state is constant C/haplength, other boundaries zero.
 * End likelihood sums M and insertion over all final-read-row columns. */
template <typename T> T scalar_scaled(const phmm_read &r, const phmm_haplotype &h,
                                    const Tables<T> &table, T initial_constant) {
    const size_t columns = static_cast<size_t>(h.length) + 1;
    std::vector<T> pm(columns, 0), pi(columns, 0), pd(columns, initial_constant / static_cast<T>(h.length));
    std::vector<T> cm(columns, 0), ci(columns, 0), cd(columns, 0);
    for (uint32_t row = 0; row < r.length; ++row) {
        cm[0] = ci[0] = cd[0] = 0;
        const unsigned iq = r.insertion_gop[row] & 127u, dq = r.deletion_gop[row] & 127u;
        const T pmm = table.mm[mm_index(iq, dq)];
        const T ins = table.phred[iq], del = table.phred[dq];
        const T gap = table.phred[r.overall_gcp[row] & 127u], gapm = T(1) - gap;
        const T err = table.phred[r.quals[row] & 127u];
        for (uint32_t column = 1; column <= h.length; ++column) {
            const uint8_t rb = r.bases[row], hb = h.bases[column - 1];
            const T emission = (rb == hb || rb == 'N' || hb == 'N') ? T(1) - err : err / T(3);
            cm[column] = ((pm[column - 1] * pmm + pi[column - 1] * gapm) + pd[column - 1] * gapm) * emission;
            ci[column] = pm[column] * ins + pi[column] * gap;
            cd[column] = cm[column - 1] * del + cd[column - 1] * gap;
        }
        pm.swap(cm); pi.swap(ci); pd.swap(cd);
    }
    T sum_m = 0, sum_i = 0;
    for (uint32_t column = 1; column <= h.length; ++column) {
        sum_m += pm[column]; sum_i += pi[column];
    }
    return sum_m + sum_i;
}
inline double cpu_double(const phmm_read &r, const phmm_haplotype &h) {
    const double initial = std::ldexp(1.0, 1020);
    const double result = scalar_scaled(r, h, double_tables(), initial);
    if (std::isnan(result) || result < 0 || std::isinf(result))
        throw std::runtime_error("CPU double PairHMM produced an invalid scaled probability");
    return std::log10(result) - std::log10(initial);
}
}
