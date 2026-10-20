#include "pairhmm_internal.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>

namespace {
struct DeviceRead { uint64_t offset; uint32_t length; uint32_t reserved; };
struct DeviceHap { uint64_t offset; uint32_t length; uint32_t reserved; };
constexpr unsigned THREADS = 128;
constexpr float MIN_ACCEPTED = 1e-28f;
void checked(cudaError_t result, const char *where) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(result));
}
uint64_t add_bytes(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("PairHMM allocation size overflow");
    return a + b;
}
template <typename T> void grow(T *&ptr, size_t &capacity, size_t required, uint64_t &owned, uint64_t &peak) {
    if (required <= capacity) return;
    if (ptr) { checked(cudaFree(ptr), "cudaFree old buffer"); owned -= capacity; ptr = nullptr; capacity = 0; }
    checked(cudaMalloc(reinterpret_cast<void **>(&ptr), required), "cudaMalloc buffer");
    capacity = required;
    owned += required; peak = std::max(peak, owned);
}
void add_stats(phmm_stats &a, const phmm_stats &b) {
    a.compute_calls += b.compute_calls; a.input_pairs += b.input_pairs;
    a.gpu_pairs += b.gpu_pairs; a.gpu_kernel_launches += b.gpu_kernel_launches;
    a.cpu_double_fallback_pairs += b.cpu_double_fallback_pairs;
    a.gpu_dp_cells += b.gpu_dp_cells; a.h2d_bytes += b.h2d_bytes; a.d2h_bytes += b.d2h_bytes;
    a.gpu_kernel_milliseconds += b.gpu_kernel_milliseconds;
    a.host_wall_milliseconds += b.host_wall_milliseconds;
    a.device_ordinal = b.device_ordinal;
    a.owned_allocated_bytes = b.owned_allocated_bytes;
    a.owned_peak_allocated_bytes = std::max(a.owned_peak_allocated_bytes, b.owned_peak_allocated_bytes);
    a.device_free_bytes_before = b.device_free_bytes_before;
    a.device_free_bytes_after = b.device_free_bytes_after;
    a.device_total_bytes = b.device_total_bytes;
    a.invalid_gpu_result_pairs += b.invalid_gpu_result_pairs;
    a.memory_snapshot_before_valid = b.memory_snapshot_before_valid;
    a.memory_snapshot_after_valid = b.memory_snapshot_after_valid;
}

/* One block per read-haplotype pair. Three anti-diagonals of each state are
 * stored in shared memory, indexed by read row; no full DP matrix. Each cell
 * uses GKL's three-state unbanded recurrence. Host log10f removes 2^120 scaling.
 * No tensor cores or reduced-precision arithmetic are used. */
__global__ void pairhmm_wavefront(const uint8_t *bytes, const DeviceRead *reads,
                                const DeviceHap *haps, uint32_t hap_count,
                                uint64_t first_pair, uint32_t batch_pairs,
                                const float *phred, const float *match_to_match,
                                uint32_t stride, float *scaled_results) {
    const uint32_t local_pair = blockIdx.x;
    if (local_pair >= batch_pairs) return;
    const uint64_t pair = first_pair + local_pair;
    const DeviceRead r = reads[pair / hap_count];
    const DeviceHap h = haps[pair % hap_count];
    const uint8_t *rb = bytes + r.offset;
    const uint8_t *q = rb + r.length, *iq = q + r.length;
    const uint8_t *dq = iq + r.length, *gq = dq + r.length;
    const uint8_t *hb = bytes + h.offset;
    extern __shared__ float states[];
    float *m = states, *ins = m + 3 * stride, *del = ins + 3 * stride;
    for (unsigned x = threadIdx.x; x < 9 * stride; x += blockDim.x) states[x] = 0.0f;
    // Other threads may own the deletion boundary slots during the clear.
    // Complete every clear before thread 0 installs those initial values.
    __syncthreads();
    const float initial = 0x1p120f / static_cast<float>(h.length);
    if (threadIdx.x == 0) { del[0] = initial; del[stride] = initial; }
    __syncthreads();
    float final_m = 0.0f, final_ins = 0.0f;
    const uint64_t last = static_cast<uint64_t>(r.length) + h.length;
    for (uint64_t diagonal = 2; diagonal <= last; ++diagonal) {
        const unsigned now = diagonal % 3, previous = (diagonal + 2) % 3, old = (diagonal + 1) % 3;
        for (uint32_t row = threadIdx.x; row <= r.length; row += blockDim.x) {
            const int64_t column = static_cast<int64_t>(diagonal) - row;
            float mv = 0.0f, iv = 0.0f, dv = 0.0f;
            if (row == 0 && column >= 0 && column <= h.length) {
                dv = initial;
            } else if (row > 0 && column >= 1 && column <= h.length) {
                const unsigned ri = row - 1;
                const unsigned i = iq[ri] & 127u, d = dq[ri] & 127u;
                const unsigned hi = i > d ? i : d, lo = i > d ? d : i;
                const float pmm = match_to_match[hi * (hi + 1) / 2 + lo];
                const float gap = phred[gq[ri] & 127u], gapm = 1.0f - gap;
                const float error = phred[q[ri] & 127u];
                const uint8_t rbase = rb[ri], hbase = hb[column - 1];
                const float emission = (rbase == hbase || rbase == 'N' || hbase == 'N') ? 1.0f - error : error / 3.0f;
                const unsigned diag = old * stride + ri;
                mv = ((m[diag] * pmm + ins[diag] * gapm) + del[diag] * gapm) * emission;
                const unsigned above = previous * stride + ri;
                iv = m[above] * phred[i] + ins[above] * gap;
                const unsigned left = previous * stride + row;
                dv = m[left] * phred[d] + del[left] * gap;
                if (row == r.length) { final_m += mv; final_ins += iv; }
            }
            const unsigned index = now * stride + row;
            m[index] = mv; ins[index] = iv; del[index] = dv;
        }
        __syncthreads();
    }
    if (threadIdx.x == r.length % blockDim.x) scaled_results[local_pair] = final_m + final_ins;
}
}

struct phmm_cuda_context {
    int device = 0;
    uint64_t max_bytes = 0;
    uint32_t max_pairs = 0;
    size_t max_shared = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    uint8_t *bytes = nullptr;
    DeviceRead *reads = nullptr;
    DeviceHap *haps = nullptr;
    float *results = nullptr, *phred = nullptr, *mm = nullptr;
    size_t bytes_capacity = 0, reads_capacity = 0, haps_capacity = 0, result_capacity = 0;
    uint64_t owned_bytes = 0, peak_owned_bytes = 0;
    phmm_stats total{};
    ~phmm_cuda_context() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        cudaFree(bytes); cudaFree(reads); cudaFree(haps); cudaFree(results); cudaFree(phred); cudaFree(mm);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
    }
};

extern "C" const char *phmm_cuda_provider(void) { return "head-native-cuda-pairhmm-fp32-wavefront-packed-H2D-initfix-v1b"; }
extern "C" int phmm_cuda_create(int device, uint64_t max_bytes, uint32_t max_pairs,
                                phmm_cuda_context **output, char *error, size_t capacity) {
    if (output) *output = nullptr;
    try {
        if (!output || max_bytes < (1u << 20) || !max_pairs || device < 0)
            throw std::invalid_argument("Invalid context limits: require >=1MiB and nonzero max_batch_pairs");
        int count = 0; checked(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (device >= count) throw std::invalid_argument("Requested CUDA device ordinal is unavailable");
        checked(cudaSetDevice(device), "cudaSetDevice");
        std::unique_ptr<phmm_cuda_context> c(new phmm_cuda_context);
        c->device = device; c->max_bytes = max_bytes; c->max_pairs = max_pairs;
        cudaDeviceProp prop{}; checked(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        c->max_shared = prop.sharedMemPerBlock;
        if (max_pairs > static_cast<uint32_t>(prop.maxGridSize[0])) throw std::invalid_argument("max_batch_pairs exceeds the device grid limit");
        size_t free_before = 0, total_before = 0;
        if (cudaMemGetInfo(&free_before, &total_before) == cudaSuccess) {
            c->total.device_free_bytes_before = free_before; c->total.device_total_bytes = total_before;
            c->total.memory_snapshot_before_valid = 1;
        }
        checked(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "cudaStreamCreate");
        checked(cudaEventCreate(&c->begin), "cudaEventCreate begin");
        checked(cudaEventCreate(&c->end), "cudaEventCreate end");
        checked(cudaMalloc(reinterpret_cast<void **>(&c->phred), sizeof(phmm::float_tables().phred)), "cudaMalloc phred");
        c->owned_bytes += sizeof(phmm::float_tables().phred); c->peak_owned_bytes = c->owned_bytes;
        checked(cudaMalloc(reinterpret_cast<void **>(&c->mm), sizeof(phmm::float_tables().mm)), "cudaMalloc mm");
        c->owned_bytes += sizeof(phmm::float_tables().mm); c->peak_owned_bytes = c->owned_bytes;
        checked(cudaMemcpy(c->phred, phmm::float_tables().phred, sizeof(phmm::float_tables().phred), cudaMemcpyHostToDevice), "cudaMemcpy phred");
        checked(cudaMemcpy(c->mm, phmm::float_tables().mm, sizeof(phmm::float_tables().mm), cudaMemcpyHostToDevice), "cudaMemcpy mm");
        c->total.device_ordinal = device;
        c->total.owned_allocated_bytes = c->owned_bytes; c->total.owned_peak_allocated_bytes = c->peak_owned_bytes;
        size_t free_after = 0, total_after = 0;
        if (cudaMemGetInfo(&free_after, &total_after) == cudaSuccess) {
            c->total.device_free_bytes_after = free_after; c->total.device_total_bytes = total_after;
            c->total.memory_snapshot_after_valid = 1;
        }
        *output = c.release(); phmm::error_text(error, capacity, ""); return 0;
    } catch (const std::exception &e) { phmm::error_text(error, capacity, e.what()); return 1; }
}
extern "C" void phmm_cuda_destroy(phmm_cuda_context *context) { delete context; }
extern "C" int phmm_cuda_get_stats(const phmm_cuda_context *c, phmm_stats *out) {
    if (!c || !out) return 1;
    *out = c->total; return 0;
}

extern "C" int phmm_cuda_compute(phmm_cuda_context *c, const phmm_read *reads, uint32_t nr,
                                 const phmm_haplotype *haps, uint32_t nh, int use_double,
                                 double *output, phmm_stats *stats, char *error, size_t capacity) {
    phmm_stats call{};
    // Physical capacity is immutable; keep its already observed value even
    // when this call deliberately has no fresh whole-device free snapshot.
    call.device_total_bytes = c ? c->total.device_total_bytes : 0;
    // Owned allocations are counted exactly on every call. Device-wide driver
    // queries are diagnostics: sample on allocation growth and every 64 calls,
    // rather than adding two driver transactions to each tiny likelihood batch.
    bool sample_device_memory = c && ((c->total.compute_calls % 64) == 0);
    const auto started = std::chrono::steady_clock::now();
    auto finish = [&] {
        call.host_wall_milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (c) {
            call.owned_allocated_bytes = c->owned_bytes; call.owned_peak_allocated_bytes = c->peak_owned_bytes;
            size_t free_bytes = 0, total_bytes = 0;
            if (sample_device_memory && cudaSetDevice(c->device) == cudaSuccess && cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                call.device_free_bytes_after = free_bytes; call.device_total_bytes = total_bytes;
                call.memory_snapshot_after_valid = 1;
            }
            add_stats(c->total, call);
        }
        if (stats) *stats = call;
    };
    try {
        if (!c) throw std::invalid_argument("Null CUDA context");
        call.compute_calls = 1; call.device_ordinal = c->device;
        if (use_double) throw std::invalid_argument("CUDA provider implements FP32; useDoublePrecision=true is unsupported");
        const uint64_t pairs = static_cast<uint64_t>(nr) * nh;
        call.input_pairs = pairs;
        if (!pairs) { finish(); phmm::error_text(error, capacity, ""); return 0; }
        if (!reads || !haps || !output || pairs > SIZE_MAX / sizeof(double))
            throw std::invalid_argument("Null batch argument or overflowing output length");
        checked(cudaSetDevice(c->device), "cudaSetDevice compute");
        uint32_t maximum_read = 0;
        uint64_t packed_size = 0;
        for (uint32_t r = 0; r < nr; ++r) {
            const auto &read = reads[r];
            if (!read.length || read.length > phmm::MAX_GPU_READ_LENGTH || !read.bases || !read.quals ||
                !read.insertion_gop || !read.deletion_gop || !read.overall_gcp)
                throw std::invalid_argument("CUDA read length must be 1..1024 and all Phred arrays must be present");
            for (uint32_t i = 0; i < read.length; ++i)
                if (!phmm::valid_base(read.bases[i])) throw std::invalid_argument("Read base outside A/C/G/T/N");
            maximum_read = std::max(maximum_read, read.length);
            packed_size = add_bytes(packed_size, static_cast<uint64_t>(read.length) * 5);
        }
        for (uint32_t h = 0; h < nh; ++h) {
            if (!haps[h].length || !haps[h].bases || haps[h].length > INT32_MAX - phmm::MAX_GPU_READ_LENGTH)
                throw std::invalid_argument("Invalid haplotype length or null bases");
            for (uint32_t j = 0; j < haps[h].length; ++j)
                if (!phmm::valid_base(haps[h].bases[j])) throw std::invalid_argument("Haplotype base outside A/C/G/T/N");
            packed_size = add_bytes(packed_size, haps[h].length);
        }
        const uint32_t stride = maximum_read + 1;
        const size_t shared_bytes = static_cast<size_t>(9) * stride * sizeof(float);
        if (shared_bytes > c->max_shared) throw std::invalid_argument("Read length exceeds CUDA shared-memory capacity");
        const uint32_t limit = static_cast<uint32_t>(std::min<uint64_t>(c->max_pairs, pairs));
        const size_t read_bytes = static_cast<size_t>(nr) * sizeof(DeviceRead);
        const size_t hap_bytes = static_cast<size_t>(nh) * sizeof(DeviceHap);
        // Metadata and bases share one arena and one transfer. Both metadata
        // structures are 16 bytes; cudaMalloc provides the required alignment.
        const uint64_t data_offset = add_bytes(read_bytes, hap_bytes);
        packed_size = add_bytes(packed_size, data_offset);
        const size_t result_bytes = static_cast<size_t>(limit) * sizeof(float);
        const bool needs_growth = packed_size > c->bytes_capacity ||
            result_bytes > c->result_capacity;
        sample_device_memory = sample_device_memory || needs_growth;
        if (sample_device_memory) {
            size_t free_before = 0, total_before = 0;
            if (cudaMemGetInfo(&free_before, &total_before) == cudaSuccess) {
                call.device_free_bytes_before = free_before;
                call.device_total_bytes = total_before;
                call.memory_snapshot_before_valid = 1;
            }
        }
        uint64_t owned = sizeof(phmm::float_tables().phred) + sizeof(phmm::float_tables().mm);
        owned = add_bytes(owned, std::max<uint64_t>(packed_size, c->bytes_capacity));
        owned = add_bytes(owned, std::max(result_bytes, c->result_capacity));
        if (owned > c->max_bytes) throw std::runtime_error("Batch input and retained buffers exceed context-owned CUDA byte limit");
        if (packed_size > SIZE_MAX) throw std::overflow_error("Packed batch exceeds host address space");
        std::vector<uint8_t> packed(static_cast<size_t>(packed_size));
        std::vector<DeviceRead> read_meta(nr);
        std::vector<DeviceHap> hap_meta(nh);
        size_t offset = static_cast<size_t>(data_offset);
        for (uint32_t r = 0; r < nr; ++r) {
            const auto &read = reads[r]; read_meta[r] = {offset, read.length, 0};
            for (const uint8_t *array : {read.bases, read.quals, read.insertion_gop, read.deletion_gop, read.overall_gcp}) {
                std::memcpy(packed.data() + offset, array, read.length); offset += read.length;
            }
        }
        for (uint32_t h = 0; h < nh; ++h) {
            hap_meta[h] = {offset, haps[h].length, 0};
            std::memcpy(packed.data() + offset, haps[h].bases, haps[h].length); offset += haps[h].length;
        }
        std::memcpy(packed.data(), read_meta.data(), read_bytes);
        std::memcpy(packed.data() + read_bytes, hap_meta.data(), hap_bytes);
        grow(c->bytes, c->bytes_capacity, packed.size(), c->owned_bytes, c->peak_owned_bytes);
        grow(c->results, c->result_capacity, result_bytes, c->owned_bytes, c->peak_owned_bytes);
        checked(cudaMemcpyAsync(c->bytes, packed.data(), packed.size(), cudaMemcpyHostToDevice, c->stream), "H2D packed metadata, bases and Phreds");
        call.h2d_bytes = packed.size();
        std::vector<float> results(limit);
        const float log10_initial = ::log10f(std::ldexp(1.0f, 120));
        for (uint64_t first = 0; first < pairs; first += limit) {
            const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(limit, pairs - first));
            checked(cudaEventRecord(c->begin, c->stream), "record kernel begin");
            pairhmm_wavefront<<<count, THREADS, shared_bytes, c->stream>>>(c->bytes,
                reinterpret_cast<const DeviceRead*>(c->bytes),
                reinterpret_cast<const DeviceHap*>(c->bytes + read_bytes),
                nh, first, count, c->phred, c->mm, stride, c->results);
            checked(cudaGetLastError(), "launch PairHMM wavefront");
            ++call.gpu_kernel_launches;
            checked(cudaEventRecord(c->end, c->stream), "record kernel end");
            checked(cudaMemcpyAsync(results.data(), c->results, count * sizeof(float), cudaMemcpyDeviceToHost, c->stream), "D2H scaled likelihoods");
            checked(cudaStreamSynchronize(c->stream), "synchronize PairHMM batch");
            float milliseconds = 0; checked(cudaEventElapsedTime(&milliseconds, c->begin, c->end), "kernel elapsed time");
            call.gpu_kernel_milliseconds += milliseconds; call.gpu_pairs += count;
            call.d2h_bytes += count * sizeof(float);
            uint64_t first_invalid = UINT64_MAX;
            for (uint32_t p = 0; p < count; ++p) {
                const uint64_t pair = first + p;
                const uint32_t r = pair / nh, h = pair % nh;
                call.gpu_dp_cells += static_cast<uint64_t>(reads[r].length) * haps[h].length;
                if (!std::isfinite(results[p]) || results[p] < 0.0f) {
                    ++call.invalid_gpu_result_pairs;
                    if (first_invalid == UINT64_MAX) first_invalid = pair;
                }
            }
            if (first_invalid != UINT64_MAX)
                throw std::runtime_error("GPU PairHMM produced NaN/Inf/negative probability at pair index " + std::to_string(first_invalid));
            for (uint32_t p = 0; p < count; ++p) {
                const uint64_t pair = first + p;
                const uint32_t r = pair / nh, h = pair % nh;
                const float scaled = results[p];
                if (scaled < MIN_ACCEPTED) {
                    ++call.cpu_double_fallback_pairs; output[pair] = phmm::cpu_double(reads[r], haps[h]);
                } else output[pair] = static_cast<double>(::log10f(scaled) - log10_initial);
            }
        }
        finish(); phmm::error_text(error, capacity, ""); return 0;
    } catch (const std::exception &e) {
        sample_device_memory = true; // Keep fresh device evidence on failure.
        if (c && c->stream) cudaStreamSynchronize(c->stream);
        finish(); phmm::error_text(error, capacity, e.what()); return 1;
    }
}
