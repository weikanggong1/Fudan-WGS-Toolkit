# Source and license

The adapter and comparator are project code under the repository GPL-3.0 license.
The comparator reuses the project's complete-record/report comparison code.
No GATK, HTSJDK or Picard binary or upstream implementation is bundled. No
reference sequence, known-site resource, raw read, participant identity or locus
output is included.

The separately installed Conda GATK 4.2.0.0 jar provides the unchanged algorithms
and readers. The exact tag's [LICENSE.TXT](https://github.com/broadinstitute/gatk/blob/4.2.0.0/LICENSE.TXT)
is Apache-2.0, SHA-256
`0e53509666a2480048ef7ae7dfdfc24a075fc1d6492295f5fa93aa60eb7d226a`.
HTSJDK is MIT; the installed dependencies' applicable third-party notices remain
in force. Users obtain these tools through the documented Conda environment.
This is an execution adapter for the original BQSR algorithm.

The actual fixed jar APIs were checked with `javap`. ReadWalker makes each
FeatureContext from the original unmodified read interval. An unmapped or invalid
interval is null and yields an empty known-site result. The producer carries
that immutable interval into its queued job; each worker creates a FeatureContext
with its own original FeatureManager and queries the same ordered FeatureInput
collection before the original engine transforms its private read copy.

All worker managers are constructed sequentially before threads start, retaining
the original cloud-buffer values, GenomicsDB options and 1000-base lookahead.
Each contiguous group of 16 eligible reads is assigned to one worker, then the
next group goes to the next worker. This fixed internal scheduling policy
improves reader locality while retaining every eligible read and original read
interval. It is not a new user parameter or a change to the known-site window.
Each worker also owns its engine, covariates, reference reader, header and argument
copy. Reference and feature readers are closed independently; failure propagates,
and cleanup also covers managers whose threads never started. No live
FeatureContext or mutable feature cache crosses worker threads.

The standard original training predicate is initialized with the same header.
The inherited pre-filter and post-filter read transformers are identity in this
fixed GATK version; original BaseRecalibrator does not override them. The public
command plan retains default BAQ=false and indel-BQSR=false and supplies no
interval, downsampling or filter-changing options. All decoded records go to
the replay writer, including records excluded by the training predicate.

Workers return unfinalized original in-memory RecalibrationTables. They are
combined in worker-index order with original inPlaceCombine; original global
finalization, rounding, quantization and report construction each run once.
The temporary BAM defaults to compression 0, with a bounded 256-record writer
buffer. The original ApplyBQSR and its final compression settings are retained.
Whole-route timing includes the temporary write and ApplyBQSR; queue capacity is
a count bound, while the resource runner separately measures process-tree PSS.
The current project Java source is SHA-256
`8aa8f9a6f9846e51eb21e145c624f74962baf948f0fe4a489251bb6831f51465`.
The same source passed the real 1M paired-read-prefix component test and a later
10M paired-read-prefix public-API comparison with cold public-builder artifacts.
All five report tables and five complete alignment-record groups matched; each
group covered 19,983,249 records per path, including QUAL, typed AUX, tag order
and record order. The complete VCF and independent PLINK readbacks also passed.
The API comparison used 44 prebound 200 kb calling windows, not full-depth WGS.
Timing, resource coverage and limitations are reported in the
[entrypoint benchmark](../../docs/single_subj_preprocessing.md#99-v012更大真实输入的同配置完整调用).

Upstream references fixed to 4.2.0.0:

- [BaseRecalibrator](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/tools/walkers/bqsr/BaseRecalibrator.java): original filters, engine and report construction.
- [ReadWalker](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/engine/ReadWalker.java): traversal and original read-interval semantics.
- [FeatureContext](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/engine/FeatureContext.java) and [FeatureManager](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/engine/FeatureManager.java): original ordered resource queries and isolated reader lifecycle.
- [FeatureCache](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/engine/FeatureCache.java): original 1000-base prefetch, interval overlap and contig/refill behavior; unchanged here.
- [BaseRecalibrationEngine](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/utils/recalibration/BaseRecalibrationEngine.java) and [RecalibrationTables](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/utils/recalibration/RecalibrationTables.java): original per-read statistics, unfinalized-table combination and global finalization.
- [ApplyBQSR](https://github.com/broadinstitute/gatk/blob/4.2.0.0/src/main/java/org/broadinstitute/hellbender/tools/walkers/bqsr/ApplyBQSR.java): original second-pass quality transformation.
- [HTSJDK 2.24.0 SAMFileWriterFactory](https://github.com/samtools/htsjdk/blob/2.24.0/src/main/java/htsjdk/samtools/SAMFileWriterFactory.java): bounded asynchronous temporary BAM writer. GATK's tagged build declares HTSJDK 2.24.0.

References: McKenna et al., Genome Research (2010),
[doi:10.1101/gr.107524.110](https://doi.org/10.1101/gr.107524.110);
DePristo et al., Nature Genetics (2011),
[doi:10.1038/ng.806](https://doi.org/10.1038/ng.806).
