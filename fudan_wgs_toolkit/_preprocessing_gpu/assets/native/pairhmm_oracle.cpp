#include <type_traits>
#include "pairhmm_internal.hpp"
extern "C" int phmm_cpu_double_log10(const phmm_read *r, const phmm_haplotype *h,
                                     double *out, char *error, size_t capacity) {
    try {
        if (!r || !h || !out) throw std::invalid_argument("Null oracle argument");
        phmm::validate(*r, *h);
        *out = phmm::cpu_double(*r, *h);
        phmm::error_text(error, capacity, "");
        return 0;
    } catch (const std::exception &e) {
        phmm::error_text(error, capacity, e.what()); return 1;
    }
}
