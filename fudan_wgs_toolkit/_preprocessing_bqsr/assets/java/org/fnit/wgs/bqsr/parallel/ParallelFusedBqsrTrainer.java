package org.fnit.wgs.bqsr.parallel;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import htsjdk.samtools.SAMFileHeader;
import htsjdk.samtools.SAMFileWriterFactory;
import htsjdk.tribble.Feature;
import java.io.*;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.ArgumentCollection;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.hellbender.engine.*;
import org.broadinstitute.hellbender.engine.filters.ReadFilter;
import org.broadinstitute.hellbender.engine.filters.ReadFilterLibrary;
import org.broadinstitute.hellbender.tools.walkers.bqsr.BaseRecalibrator;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.read.GATKRead;
import org.broadinstitute.hellbender.utils.read.SAMFileGATKReadWriter;
import org.broadinstitute.hellbender.utils.recalibration.*;
import picard.cmdline.programgroups.ReadDataManipulationProgramGroup;

/** Experimental traversal adapter, not a replacement BQSR algorithm.
 * Each worker runs the original installed GATK engine. As in the official
 * BaseRecalibratorSparkFn, unfinalized native tables are combined before the
 * one global finalization, rounding and quantization. No shard text is read.
 */
@CommandLineProgramProperties(
    summary="Bounded parallel original-GATK training with all-record replay",
    oneLineSummary="Parallel native BQSR tables, one input traversal",
    programGroup=ReadDataManipulationProgramGroup.class)
public final class ParallelFusedBqsrTrainer extends ReadWalker {
    @ArgumentCollection public final RecalibrationArgumentCollection recalArgs=new RecalibrationArgumentCollection();
    @Argument(fullName="known-sites",optional=false,doc="Original ordered known-site resources")
    public List<FeatureInput<Feature>> knownSites;
    @Argument(fullName="output",shortName="O",optional=false,doc="New recalibration report")
    public GATKPath output;
    @Argument(fullName="canonical-bam",optional=false,doc="New all-decoded-record canonical BAM")
    public String canonicalBam;
    @Argument(fullName="audit-json",optional=false,doc="New numeric completion evidence")
    public String auditJson;
    @Argument(fullName="training-workers",optional=true,doc="Original-engine worker count; 1..16")
    public int trainingWorkers=2;
    @Argument(fullName="queue-capacity",optional=true,doc="Per-worker queued reads; 1..64 and total<=256")
    public int queueCapacity=32;
    @Argument(fullName="replay-compression",optional=true,doc="Temporary BAM compression level 0..9")
    public int replayCompression=0;

    private final AtomicReference<Throwable> failure=new AtomicReference<>();
    private final List<Worker> workers=new ArrayList<>();
    private SAMFileGATKReadWriter replay;
    private ReadFilter trainingFilter;
    private long decoded,eligible,unmapped,secondary,supplementary,duplicates,vendorFailed;
    private boolean completed,workersJoined;
    private Long mainCpuStart;
    private long mainWallStart;
    private static final Job STOP=new Job(null,null);

    private static Long currentCpuNanos() {
        ThreadMXBean bean=ManagementFactory.getThreadMXBean();
        if(!bean.isCurrentThreadCpuTimeSupported() || !bean.isThreadCpuTimeEnabled())return null;
        long value=bean.getCurrentThreadCpuTime();return value<0?null:value;
    }
    private static Double cpuSeconds(Long start,Long end) {
        return start==null || end==null || end<start?null:(end-start)/1e9;
    }

    private static final class Job {
        final GATKRead read;
        final SimpleInterval originalInterval;
        Job(GATKRead read,SimpleInterval originalInterval) { this.read=read;this.originalInterval=originalInterval; }
    }

    private final class Worker implements Runnable {
        final int index;
        final ArrayBlockingQueue<Job> queue=new ArrayBlockingQueue<>(queueCapacity);
        final SAMFileHeader header;
        final RecalibrationArgumentCollection args;
        final Thread thread;
        final FeatureManager featureManager;
        boolean closedFeatureManager;
        BaseRecalibrationEngine engine;
        long submittedCalls;
        int maximumQueued;
        boolean closedReference;
        Double cpuSeconds,wallSeconds;
        Worker(int index) {
            this.index=index;header=getHeaderForReads().clone();args=copyArguments(recalArgs);
            // Construct sequentially on the producer before starting threads:
            // FeatureInput codec metadata is initialized once without races.
            featureManager=new FeatureManager(ParallelFusedBqsrTrainer.this,ReadWalker.FEATURE_CACHE_LOOKAHEAD,
                cloudPrefetchBuffer,cloudIndexPrefetchBuffer,getGenomicsDBOptions());
            thread=new Thread(this,"FNIT-original-BQSR-"+index);
        }
        synchronized void closeOwnedManager() {
            if(closedFeatureManager)return;
            featureManager.close();closedFeatureManager=true;
        }
        @Override public void run() {
            final long wallStart=System.nanoTime();
            final Long cpuStart=currentCpuNanos();
            ReferenceDataSource reference=null;
            try {
                engine=new BaseRecalibrationEngine(args,header);
                reference=ReferenceDataSource.of(referenceArguments.getReferencePath());
                while(true) {
                    Job job=queue.take();
                    if(job==STOP)break;
                    if(failure.get()!=null)break;
                    // Query the same original interval and ordered known-site
                    // inputs with this worker's private original GATK manager.
                    FeatureContext featureContext=new FeatureContext(featureManager,job.originalInterval);
                    engine.processRead(job.read,reference,featureContext.getValues(knownSites));
                    submittedCalls++;
                }
            } catch(Throwable error) {
                failure.compareAndSet(null,error);
            } finally {
                if(reference!=null) {
                    try { reference.close();closedReference=true; }
                    catch(Throwable error) { failure.compareAndSet(null,error); }
                }
                try { closeOwnedManager(); }
                catch(Throwable error) { failure.compareAndSet(null,error); }
                cpuSeconds=cpuSeconds(cpuStart,currentCpuNanos());
                wallSeconds=(System.nanoTime()-wallStart)/1e9;
            }
        }
    }

    private static RecalibrationArgumentCollection copyArguments(RecalibrationArgumentCollection original) {
        // The fixed GATK class is Serializable. Each engine owns its argument
        // instance and header, including platform defaults and covariate state.
        try {
            ByteArrayOutputStream bytes=new ByteArrayOutputStream();
            try(ObjectOutputStream stream=new ObjectOutputStream(bytes)) { stream.writeObject(original); }
            try(ObjectInputStream stream=new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
                return (RecalibrationArgumentCollection)stream.readObject();
            }
        } catch(IOException|ClassNotFoundException error) { throw new IllegalStateException("Cannot copy original BQSR arguments",error); }
    }

    @Override public boolean requiresReference() { return true; }
    @Override public List<ReadFilter> getDefaultReadFilters() {
        return Collections.singletonList(ReadFilterLibrary.ALLOW_ALL_READS);
    }

    @Override public void onTraversalStart() {
        mainCpuStart=currentCpuNanos();mainWallStart=System.nanoTime();
        if(trainingWorkers<1 || trainingWorkers>16 || queueCapacity<1 || queueCapacity>64
                || trainingWorkers*queueCapacity>256)
            throw new IllegalArgumentException("Use 1..16 workers with 1..64 queued reads each and <=256 queued reads in total");
        if(trainingWorkers>Runtime.getRuntime().availableProcessors())
            throw new IllegalArgumentException("Worker count exceeds the JVM's active processor budget");
        if(replayCompression<0 || replayCompression>9)throw new IllegalArgumentException("Compression must be 0..9");
        if(recalArgs.enableBAQ)throw new IllegalArgumentException("This candidate is validated only for original default BAQ=false");
        if(hasUserSuppliedIntervals())throw new IllegalArgumentException("Full input traversal is required; interval filtering is unsupported");
        if(Files.exists(Paths.get(canonicalBam)) || Files.exists(Paths.get(auditJson)) || Files.exists(Paths.get(output.toString())))
            throw new IllegalArgumentException("All output paths must be new");
        if(recalArgs.FORCE_PLATFORM!=null)recalArgs.DEFAULT_PLATFORM=recalArgs.FORCE_PLATFORM;
        Utils.warnOnNonIlluminaReadGroups(getHeaderForReads(),logger);
        trainingFilter=ReadFilter.fromList(BaseRecalibrator.getStandardBQSRReadFilterList(),getHeaderForReads());
        SAMFileWriterFactory factory=new SAMFileWriterFactory().setCompressionLevel(replayCompression)
            .setCreateIndex(false).setUseAsyncIo(true).setAsyncOutputBufferSize(256);
        replay=new SAMFileGATKReadWriter(factory.makeBAMWriter(getHeaderForReads().clone(),true,Paths.get(canonicalBam)));
        for(int i=0;i<trainingWorkers;i++)workers.add(new Worker(i));
        for(Worker worker:workers)worker.thread.start();
    }

    private void checkFailure() {
        Throwable error=failure.get();
        if(error!=null)throw new IllegalStateException("Original-engine worker failed",error);
    }

    private void offer(Worker worker,Job job) {
        try {
            while(true) {
                checkFailure();
                if(!worker.thread.isAlive())throw new IllegalStateException("BQSR worker exited before its queue was drained");
                if(worker.queue.offer(job,250,TimeUnit.MILLISECONDS)) {
                    worker.maximumQueued=Math.max(worker.maximumQueued,worker.queue.size());
                    checkFailure();return;
                }
            }
        } catch(InterruptedException error) {
            Thread.currentThread().interrupt();throw new IllegalStateException("BQSR producer was interrupted",error);
        }
    }

    @Override public void apply(GATKRead read,ReferenceContext reference,FeatureContext features) {
        checkFailure();
        // Deep copy before the asynchronous replay writer can touch lazy SAM
        // fields. Only its immutable interval crosses to a worker; no live
        // FeatureContext or mutable feature cache is shared across threads.
        final GATKRead trainingRead=read.copy();
        replay.addRead(read);decoded++;
        if(read.isUnmapped())unmapped++;
        if(read.isSecondaryAlignment())secondary++;
        if(read.isSupplementaryAlignment())supplementary++;
        if(read.isDuplicate())duplicates++;
        if(read.failsVendorQualityCheck())vendorFailed++;
        if(trainingFilter.test(trainingRead)) {
            // Carry the immutable interval already made by ReadWalker before
            // any engine clipping; do not recompute it from a transformed read.
            final SimpleInterval originalInterval=features.getInterval();
            Worker worker=workers.get((int)((eligible/16)%trainingWorkers));
            offer(worker,new Job(trainingRead,originalInterval));eligible++;
        }
    }

    private void joinWorkers(boolean cancel,long timeoutSeconds) {
        if(workersJoined)return;
        if(cancel)for(Worker worker:workers)worker.thread.interrupt();
        final long deadline=System.nanoTime()+TimeUnit.SECONDS.toNanos(timeoutSeconds);
        boolean interrupted=false;
        for(Worker worker:workers) {
            while(worker.thread.isAlive()) {
                long remaining=deadline-System.nanoTime();
                if(remaining<=0)break;
                try { worker.thread.join(Math.max(1,TimeUnit.NANOSECONDS.toMillis(remaining))); }
                catch(InterruptedException error) { interrupted=true;for(Worker other:workers)other.thread.interrupt(); }
            }
        }
        if(interrupted)Thread.currentThread().interrupt();
        for(Worker worker:workers)if(worker.thread.isAlive())
            throw new IllegalStateException("BQSR worker did not terminate within the bounded shutdown interval");
        workersJoined=true;
        for(Worker worker:workers)worker.queue.clear();
    }

    @Override public Object onTraversalSuccess() {
        replay.close();replay=null;
        for(Worker worker:workers)offer(worker,STOP);
        joinWorkers(false,300);checkFailure();
        long processed=0,calls=0;
        for(Worker worker:workers) { calls+=worker.submittedCalls;processed+=worker.engine.getNumReadsProcessed(); }
        if(calls!=eligible)throw new IllegalStateException("A submitted training read was lost or duplicated");
        RecalibrationTables combined=workers.get(0).engine.getRecalibrationTables();
        for(int i=1;i<workers.size();i++)
            RecalibrationTables.inPlaceCombine(combined,workers.get(i).engine.getRecalibrationTables());
        BaseRecalibrationEngine.finalizeRecalibrationTables(combined);
        QuantizationInfo quantized=new QuantizationInfo(combined,recalArgs.QUANTIZING_LEVELS);
        try(PrintStream stream=new PrintStream(output.getOutputStream())) {
            RecalUtils.outputRecalibrationReport(stream,recalArgs,quantized,combined,workers.get(0).engine.getCovariates());
            if(stream.checkError())throw new IllegalStateException("Recalibration report write failed");
        }
        JsonObject evidence=new JsonObject();
        evidence.addProperty("status","original_GATK_parallel_training_and_all_records_replay_complete");
        evidence.addProperty("all_decoded_records",decoded);
        evidence.addProperty("standard_training_filter_passed_records",eligible);
        evidence.addProperty("original_engine_calls",calls);
        evidence.addProperty("original_engine_processed_records",processed);
        evidence.addProperty("unmapped_records",unmapped);evidence.addProperty("secondary_records",secondary);
        evidence.addProperty("supplementary_records",supplementary);evidence.addProperty("duplicate_records",duplicates);
        evidence.addProperty("vendor_failed_records",vendorFailed);evidence.addProperty("temporary_BAM_compression",replayCompression);
        evidence.addProperty("training_workers",trainingWorkers);evidence.addProperty("per_worker_queue_capacity",queueCapacity);
        evidence.addProperty("maximum_queued_read_capacity",trainingWorkers*queueCapacity);
        evidence.addProperty("writer_buffer_read_capacity",256);
        evidence.addProperty("queued_records_are_a_count_bound_not_a_byte_bound",true);
        evidence.addProperty("worker_private_engine_reference_header_and_arguments",true);
        evidence.addProperty("BAQ_enabled",recalArgs.enableBAQ);
        evidence.addProperty("worker_tables_finalized_before_merge",false);
        evidence.addProperty("global_finalization_and_quantization_count",1);
        evidence.addProperty("all_records_written_before_training_filter",true);
        evidence.addProperty("writer_closed_and_flushed",true);
        evidence.addProperty("all_workers_joined_and_references_closed",true);
        evidence.addProperty("raw_read_identifiers_or_sequences_exported",false);
        evidence.addProperty("GPU_work_started",false);
        evidence.addProperty("FeatureManager_lookahead_bases",ReadWalker.FEATURE_CACHE_LOOKAHEAD);
        evidence.addProperty("known_site_queries_in_workers_with_private_FeatureManager",true);
        evidence.addProperty("eligible_dispatch_contiguous_chunk_size",16);
        evidence.addProperty("worker_FeatureManagers_constructed_sequentially_before_start",true);
        evidence.addProperty("queued_interval_is_original_ReadWalker_FeatureContext_interval",true);
        evidence.addProperty("producer_main_thread_cpu_seconds",cpuSeconds(mainCpuStart,currentCpuNanos()));
        evidence.addProperty("producer_main_thread_wall_seconds",(System.nanoTime()-mainWallStart)/1e9);
        evidence.addProperty("thread_times_scope","main:traversal_start_through_final_report;workers:start_through_reference_close_including_queue_wait");
        evidence.addProperty("thread_CPU_is_not_added_to_task_CPU",true);
        JsonArray counts=new JsonArray();
        for(Worker worker:workers) {
            if(!worker.closedReference || !worker.closedFeatureManager)
                throw new IllegalStateException("A worker reference or feature manager was not closed");
            JsonObject row=new JsonObject();row.addProperty("index",worker.index);
            row.addProperty("calls",worker.submittedCalls);row.addProperty("maximum_sampled_queued",worker.maximumQueued);counts.add(row);
            row.addProperty("thread_cpu_seconds",worker.cpuSeconds);row.addProperty("wall_seconds",worker.wallSeconds);
        }
        evidence.add("workers",counts);
        try {
            Files.write(Paths.get(auditJson),(new GsonBuilder().setPrettyPrinting().create().toJson(evidence)+"\n")
                .getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE_NEW);
        } catch(IOException error) { throw new UncheckedIOException(error); }
        completed=true;return "SUCCESS";
    }

    @Override public void closeTool() {
        Throwable problem=null;
        try { joinWorkers(true,30); }catch(Throwable error) { problem=error; }
        // Also cover managers constructed before a later constructor/start
        // failed: an unstarted thread never enters its run/finally block.
        for(Worker worker:workers)if(!worker.thread.isAlive()) {
            try { worker.closeOwnedManager(); }
            catch(Throwable error) { if(problem==null)problem=error;else problem.addSuppressed(error); }
        }
        try { if(replay!=null) { replay.close();replay=null; } }
        catch(Throwable error) { if(problem==null)problem=error;else problem.addSuppressed(error); }
        if(problem!=null)throw new IllegalStateException("BQSR shutdown was incomplete",problem);
    }

    public static void main(String[] args) {
        ParallelFusedBqsrTrainer tool=new ParallelFusedBqsrTrainer();
        try {
            tool.instanceMain(args);
            boolean informational=false;
            for(String arg:args)if(arg.equals("--help")||arg.equals("-h")||arg.equals("--version"))informational=true;
            if(!informational && (!tool.completed || tool.replay!=null || !tool.workersJoined))
                throw new IllegalStateException("Training did not complete and flush all records");
            System.exit(0);
        } catch(Throwable error) {
            try { tool.closeTool(); }catch(Throwable cleanup) { error.addSuppressed(cleanup); }
            error.printStackTrace(System.err);System.exit(1);
        }
    }
}
