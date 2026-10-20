#ifndef HEAD_PAIRHMM_CUDA_H
#define HEAD_PAIRHMM_CUDA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Bases are uppercase ASCII A/C/G/T/N; qualities are raw unsigned Phred bytes,
 * never FASTQ ASCII. Arrays belong to the caller and remain live through compute.
 * This v1 interface rejects zero lengths and bases outside this alphabet. */
typedef struct {
    const uint8_t *bases;
    const uint8_t *quals;
    const uint8_t *insertion_gop;
    const uint8_t *deletion_gop;
    const uint8_t *overall_gcp;
    uint32_t length;
} phmm_read;
typedef struct { const uint8_t *bases; uint32_t length; } phmm_haplotype;
typedef struct phmm_cuda_context phmm_cuda_context;

typedef struct {
    uint64_t compute_calls;
    uint64_t input_pairs;
    uint64_t gpu_pairs;
    uint64_t gpu_kernel_launches;
    uint64_t cpu_double_fallback_pairs;
    uint64_t gpu_dp_cells;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    double gpu_kernel_milliseconds;
    double host_wall_milliseconds;
    int32_t device_ordinal;
    int32_t reserved;
    uint64_t owned_allocated_bytes;
    uint64_t owned_peak_allocated_bytes;
    uint64_t device_free_bytes_before;
    uint64_t device_free_bytes_after;
    uint64_t device_total_bytes;
    uint64_t invalid_gpu_result_pairs;
    uint32_t memory_snapshot_before_valid;
    uint32_t memory_snapshot_after_valid;
} phmm_stats;

/* No silent CPU provider: create/compute return nonzero on any CUDA failure.
 * max_device_bytes bounds allocations owned by this context (minimum 1 MiB).
 * max_batch_pairs bounds simultaneous blocks/results; memory may reduce it.
 * use_double_precision != 0 is rejected; low FP32 likelihoods are explicitly
 * counted CPU-double fallbacks following GKL 0.8.6 MIN_ACCEPTED=1e-28f.
 * GPU NaN/Inf/negative probabilities are fatal and separately counted.
 * Memory snapshots are whole-device observations; valid=0 means unknown.
 * owned fields count successful cudaMalloc buffers, excluding driver overhead. */
int phmm_cuda_create(int device_ordinal, uint64_t max_device_bytes,
                     uint32_t max_batch_pairs, phmm_cuda_context **out_context,
                     char *error, size_t error_capacity);
int phmm_cuda_compute(phmm_cuda_context *context,
                     const phmm_read *reads, uint32_t num_reads,
                     const phmm_haplotype *haplotypes, uint32_t num_haplotypes,
                     int use_double_precision, double *log10_likelihoods,
                     phmm_stats *call_stats, char *error, size_t error_capacity);
void phmm_cuda_destroy(phmm_cuda_context *context);
/* Returns cumulative successful and partial-call counters for this context. */
int phmm_cuda_get_stats(const phmm_cuda_context *context, phmm_stats *stats);
const char *phmm_cuda_provider(void);

/* Independent scalar, row-major CPU oracle; double recurrence and GKL 0.8.6
 * double transition/probability tables, 2^1020 initialization, log10 output.
 * It is not a GPU implementation and does not increment GPU counters. */
int phmm_cpu_double_log10(const phmm_read *read, const phmm_haplotype *haplotype,
                         double *result, char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
