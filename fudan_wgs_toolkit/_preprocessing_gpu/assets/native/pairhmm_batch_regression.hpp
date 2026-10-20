#pragma once
#include "pairhmm_internal.hpp"
#include <array>
#include <string>

struct BatchRegressionResult {
    uint64_t calls = 0, pairs = 0, kernel_launches = 0;
    uint64_t short_pairs = 0, long_pairs = 0;
    double max_scalar_error = 0, max_double_error = 0;
};

// A correctness regression, never a scientific performance benchmark. Large
// simultaneous grids exposed the old shared-zeroing/boundary-seeding race.
inline BatchRegressionResult batch_initialization_regression() {
    BatchRegressionResult result;
    for (uint32_t length : {41u, 161u}) {
        std::array<std::string, 2> bases, hap_bases;
        std::array<std::array<std::vector<uint8_t>, 4>, 2> quality;
        std::array<phmm_read, 2> unique_reads;
        std::array<phmm_haplotype, 2> haps;
        for (unsigned r = 0; r < 2; ++r) {
            for (uint32_t i = 0; i < length; ++i) bases[r] += "ACGT"[(i * 13 + i / 7) % 4];
            hap_bases[r] = bases[r];
            if (r == 1) hap_bases[r][length / 2] = 'N';
            quality[r][0].resize(length); quality[r][1].resize(length);
            quality[r][2].resize(length); quality[r][3].resize(length);
            for (uint32_t i = 0; i < length; ++i) {
                quality[r][0][i] = 20 + (i + r) % 20;
                quality[r][1][i] = 30 + (i + r) % 20;
                quality[r][2][i] = 35 + (i + 2 * r) % 20;
                quality[r][3][i] = 8 + (i + r) % 15;
            }
            unique_reads[r] = {reinterpret_cast<const uint8_t *>(bases[r].data()),
                quality[r][0].data(), quality[r][1].data(), quality[r][2].data(),
                quality[r][3].data(), length};
            haps[r] = {reinterpret_cast<const uint8_t *>(hap_bases[r].data()), length};
        }
        std::vector<phmm_read> reads;
        for (unsigned repeat = 0; repeat < 512; ++repeat)
            for (const auto &read : unique_reads) reads.push_back(read);
        std::array<double, 4> scalar, exact;
        for (unsigned r = 0; r < 2; ++r) for (unsigned h = 0; h < 2; ++h) {
            const unsigned index = r * 2 + h;
            const float probability = phmm::scalar_scaled(unique_reads[r], haps[h],
                phmm::float_tables(), 0x1p120f);
            exact[index] = phmm::cpu_double(unique_reads[r], haps[h]);
            scalar[index] = probability < 1e-28f ? exact[index] :
                static_cast<double>(::log10f(probability) - ::log10f(0x1p120f));
        }
        for (uint32_t maximum : {32u, 128u, 512u, 2048u}) {
            phmm_cuda_context *context = nullptr; char error[1024]{};
            if (phmm_cuda_create(0, 32u << 20, maximum, &context, error, sizeof(error)))
                throw std::runtime_error(error);
            try {
                std::vector<double> output(reads.size() * haps.size());
                for (unsigned repeat = 0; repeat < 3; ++repeat) {
                    phmm_stats stats{};
                    if (phmm_cuda_compute(context, reads.data(), reads.size(), haps.data(),
                        haps.size(), 0, output.data(), &stats, error, sizeof(error)))
                        throw std::runtime_error(error);
                    if (stats.gpu_pairs != output.size() || stats.invalid_gpu_result_pairs ||
                        stats.gpu_kernel_launches != (output.size() + maximum - 1) / maximum)
                        throw std::runtime_error("Dense initialization regression GPU counters failed");
                    for (size_t i = 0; i < output.size(); ++i) {
                        const double scalar_error = std::abs(output[i] - scalar[i % 4]);
                        const double double_error = std::abs(output[i] - exact[i % 4]);
                        if (!std::isfinite(output[i]) || scalar_error > 2e-5 || double_error > 8e-5)
                            throw std::runtime_error("Dense initialization regression failed at length=" +
                                std::to_string(length) + " max_batch=" + std::to_string(maximum) +
                                " pair=" + std::to_string(i));
                        result.max_scalar_error = std::max(result.max_scalar_error, scalar_error);
                        result.max_double_error = std::max(result.max_double_error, double_error);
                    }
                    ++result.calls; result.pairs += output.size();
                    result.kernel_launches += stats.gpu_kernel_launches;
                    (length <= 160 ? result.short_pairs : result.long_pairs) += output.size();
                }
            } catch (...) { phmm_cuda_destroy(context); throw; }
            phmm_cuda_destroy(context);
        }
    }
    return result;
}
