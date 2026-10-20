package org.fnit.wgs;

import com.intel.gkl.pairhmm.IntelPairHmmOMP;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import org.broadinstitute.gatk.nativebindings.pairhmm.*;

/** Bounded replay of an actual captured HC batch. No synthetic replication. */
public final class WarmPairHmmReplay {
    private static final long MAGIC = 0x464e495450484d31L;
    private static byte[] array(DataInputStream in, int n) throws Exception {
        if (n < 1 || n > 1000000) throw new IllegalArgumentException("Invalid bounded length");
        byte[] a = new byte[n]; in.readFully(a); return a;
    }
    private static String stats(PairHMMNativeBinding b) throws Exception {
        try {
            Class<?> c = b.getClass();
            while (c != null) {
                try {
                    Field f = c.getDeclaredField("context"); f.setAccessible(true);
                    Method m = c.getDeclaredMethod("statsNative", long.class); m.setAccessible(true);
                    long context=f.getLong(b);
                    return context==0 ? "null" : (String)m.invoke(null, context);
                } catch (NoSuchFieldException e) { c = c.getSuperclass(); }
            }
            return "null";
        } catch (NoSuchMethodException absent) { return "null"; }
    }
    public static void main(String[] args) throws Exception {
        if (args.length != 5) throw new IllegalArgumentException("PRIVATE.batch NEW.output_dir warmup rounds nativeThreads");
        Path fixture=Paths.get(args[0]), out=Paths.get(args[1]);
        int warmup=Integer.parseInt(args[2]), rounds=Integer.parseInt(args[3]), threads=Integer.parseInt(args[4]);
        if (warmup<0 || warmup>100 || rounds<40 || rounds>1000 || threads<1) throw new IllegalArgumentException("Replay bound");
        if (Files.size(fixture)>16L*1024*1024) throw new IllegalArgumentException("Fixture bound");
        Files.createDirectory(out);
        ReadDataHolder[] reads; HaplotypeDataHolder[] haps; long dpCells=0;
        try(DataInputStream in=new DataInputStream(Files.newInputStream(fixture))) {
            if(in.readLong()!=MAGIC) throw new IllegalArgumentException("Magic");
            int nr=in.readInt(), nh=in.readInt();
            if(nr<1 || nh<1 || (long)nr*nh>1000000) throw new IllegalArgumentException("Pair bound");
            reads=new ReadDataHolder[nr]; haps=new HaplotypeDataHolder[nh];
            for(int i=0;i<nr;i++) { int n=in.readInt(); ReadDataHolder r=new ReadDataHolder();
                r.readBases=array(in,n); r.readQuals=array(in,n); r.insertionGOP=array(in,n);
                r.deletionGOP=array(in,n); r.overallGCP=array(in,n); reads[i]=r; }
            for(int i=0;i<nh;i++) { HaplotypeDataHolder h=new HaplotypeDataHolder(); h.haplotypeBases=array(in,in.readInt()); haps[i]=h; }
            if(in.read()!=-1) throw new IllegalArgumentException("Trailing bytes");
        }
        for(ReadDataHolder r:reads) for(HaplotypeDataHolder h:haps) dpCells+=(long)r.readBases.length*h.haplotypeBases.length;
        PairHMMNativeBinding binding=new IntelPairHmmOMP();
        long t=System.nanoTime(); if(!binding.load(null)) throw new IllegalStateException("Load failed"); long loadNs=System.nanoTime()-t;
        PairHMMNativeArguments settings=new PairHMMNativeArguments(); settings.maxNumberOfThreads=threads; settings.useDoublePrecision=false;
        t=System.nanoTime(); binding.initialize(settings); long initNs=System.nanoTime()-t;
        double[] values=new double[reads.length*haps.length], first=null; double repeatMax=0;
        String initial=stats(binding), before=null, after=null; long[] times=new long[rounds], iterationTimes=new long[rounds];
        try {
            for(int i=0;i<warmup;i++) binding.computeLikelihoods(reads,haps,values);
            before=stats(binding);
            for(int i=0;i<rounds;i++) {
                long iterationStarted=System.nanoTime();
                t=System.nanoTime(); binding.computeLikelihoods(reads,haps,values); times[i]=System.nanoTime()-t;
                for(double v:values) if(!Double.isFinite(v)) throw new IllegalStateException("Nonfinite likelihood");
                if(first==null) first=values.clone();
                else for(int j=0;j<values.length;j++) repeatMax=Math.max(repeatMax,Math.abs(values[j]-first[j]));
                String record="{\"round\":"+(i+1)+",\"whole_compute_nanoseconds\":"+times[i]+",\"native\":"+stats(binding)+"}\n";
                Files.write(out.resolve("rounds.private.jsonl"),record.getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE,StandardOpenOption.APPEND);
                iterationTimes[i]=System.nanoTime()-iterationStarted;
            }
            after=stats(binding);
        } finally { binding.done(); }
        try(DataOutputStream stream=new DataOutputStream(Files.newOutputStream(out.resolve("likelihoods.private.bin"),StandardOpenOption.CREATE_NEW))) {
            stream.writeLong(MAGIC); stream.writeInt(reads.length); stream.writeInt(haps.length);
            for(double v:values) stream.writeDouble(v);
        }
        boolean cudaMapped=false;
        for(String line:Files.readAllLines(Paths.get("/proc/self/maps"),StandardCharsets.UTF_8)) {
            if(line.contains("libcuda.so") || line.contains("libcudart.so") || line.contains("libfnit_pairhmm_jni")) cudaMapped=true;
        }
        if(Boolean.getBoolean("fnit.pairhmm.replay.assertNoCuda") && cudaMapped) throw new IllegalStateException("CPU-only replay mapped a CUDA/JNI library");
        StringBuilder elapsed=new StringBuilder("[");
        for(int i=0;i<times.length;i++) { if(i>0)elapsed.append(','); elapsed.append(times[i]); } elapsed.append(']');
        StringBuilder iterationElapsed=new StringBuilder("[");
        for(int i=0;i<iterationTimes.length;i++) { if(i>0)iterationElapsed.append(','); iterationElapsed.append(iterationTimes[i]); } iterationElapsed.append(']');
        String json="{\"status\":\"real_captured_batch_warm_replay_not_biological_depth_benchmark\",\"reads\":"+reads.length+",\"haplotypes\":"+haps.length+",\"pairs_per_round\":"+values.length+",\"dp_cells_per_round\":"+dpCells+",\"warmup_rounds\":"+warmup+",\"measured_rounds\":"+rounds+",\"native_threads_requested\":"+threads+",\"cuda_library_mapped\":"+cudaMapped+",\"load_nanoseconds\":"+loadNs+",\"initialize_nanoseconds\":"+initNs+",\"whole_compute_nanoseconds\":"+elapsed+",\"whole_iteration_including_diagnostic_and_private_write_nanoseconds\":"+iterationElapsed+",\"repeat_max_abs_log10_difference\":"+repeatMax+",\"initial_native\":"+initial+",\"before_measured_native\":"+before+",\"after_measured_native\":"+after+"}";
        Files.write(out.resolve("replay.private.json"),json.getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE_NEW);
        System.out.println(json);
    }
}
