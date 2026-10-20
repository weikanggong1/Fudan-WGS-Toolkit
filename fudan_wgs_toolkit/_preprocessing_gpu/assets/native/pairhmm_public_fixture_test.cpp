#include "pairhmm_cuda.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/* Reads the seven-column public GKL 0.8.6 pairhmm-testdata format. The resource
 * itself stays outside this source tree. GKL's test loader subtracts ASCII 33
 * from all four qualities and clamps base qualities to at least 6. */
int main(int argc, char **argv) {
    uint64_t tested = 0, failed = 0, gpu_pairs = 0, launches = 0, fallback = 0;
    uint64_t peak_bytes = 0, invalid = 0;
    double max_expected_error = 0, max_double_error = 0;
    try {
        if (argc != 2) throw std::invalid_argument("Usage: pairhmm_public_fixture_test public-GKL086-pairhmm-testdata.txt");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot read public GKL fixture file");
        char error[2048]{};
        phmm_cuda_context *context = nullptr;
        if (phmm_cuda_create(0, 32u << 20, 256, &context, error, sizeof(error))) throw std::runtime_error(error);
        std::unique_ptr<phmm_cuda_context, decltype(&phmm_cuda_destroy)> guard(context, phmm_cuda_destroy);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line);
            std::string hap, read, qualities[4], trailing;
            double expected = 0;
            if (!(row >> hap >> read >> qualities[0] >> qualities[1] >> qualities[2] >> qualities[3] >> expected))
                throw std::runtime_error("Malformed public GKL fixture row");
            if (row >> trailing) throw std::runtime_error("Unexpected public GKL fixture columns");
            std::vector<uint8_t> q[4];
            for (unsigned k = 0; k < 4; ++k) {
                if (qualities[k].size() != read.size()) throw std::runtime_error("Fixture Phred/read length mismatch");
                q[k].reserve(read.size());
                for (unsigned char character : qualities[k]) {
                    if (character < 33) throw std::runtime_error("Fixture quality outside ASCII Phred+33");
                    q[k].push_back(static_cast<uint8_t>(std::max(k == 0 ? 6 : 0, static_cast<int>(character) - 33)));
                }
            }
            phmm_read r{reinterpret_cast<const uint8_t *>(read.data()), q[0].data(), q[1].data(), q[2].data(), q[3].data(), static_cast<uint32_t>(read.size())};
            phmm_haplotype h{reinterpret_cast<const uint8_t *>(hap.data()), static_cast<uint32_t>(hap.size())};
            double result = 0, oracle = 0;
            phmm_stats stats{};
            if (phmm_cuda_compute(context, &r, 1, &h, 1, 0, &result, &stats, error, sizeof(error))) throw std::runtime_error(error);
            if (phmm_cpu_double_log10(&r, &h, &oracle, error, sizeof(error))) throw std::runtime_error(error);
            ++tested; gpu_pairs += stats.gpu_pairs; launches += stats.gpu_kernel_launches;
            fallback += stats.cpu_double_fallback_pairs; invalid += stats.invalid_gpu_result_pairs;
            peak_bytes = std::max(peak_bytes, stats.owned_peak_allocated_bytes);
            const double expected_difference = std::abs(result - expected), oracle_difference = std::abs(result - oracle);
            max_expected_error = std::max(max_expected_error, expected_difference);
            max_double_error = std::max(max_double_error, oracle_difference);
            if (!std::isfinite(result) || expected_difference > 1e-5 || oracle_difference > 8e-5) ++failed;
        }
        if (!tested || gpu_pairs != tested || launches != tested || invalid) throw std::runtime_error("GPU counters do not verify every public fixture pair");
        std::cout << std::setprecision(12)
            << "{\"status\":\"" << (failed ? "failed" : "passed") << "\",\"component_only\":true,\"scientific_benchmark\":false,\"gpu_executed\":true"
            << ",\"public_GKL_version\":\"0.8.6\",\"tested_pairs\":" << tested << ",\"failed_pairs\":" << failed
            << ",\"expected_log10_threshold\":0.00001,\"double_log10_threshold\":0.00008"
            << ",\"max_gpu_expected_log10_error\":" << max_expected_error << ",\"max_gpu_double_log10_error\":" << max_double_error
            << ",\"gpu_pairs\":" << gpu_pairs << ",\"gpu_kernel_launches\":" << launches
            << ",\"cpu_double_fallback_pairs\":" << fallback << ",\"owned_peak_allocated_bytes\":" << peak_bytes
            << ",\"invalid_gpu_result_pairs\":" << invalid << "}\n";
        return failed ? 1 : 0;
    } catch (const std::exception &e) {
        std::cerr << "Public GKL GPU component test failed after " << tested << " pairs: " << e.what() << '\n';
        return 1;
    }
}
