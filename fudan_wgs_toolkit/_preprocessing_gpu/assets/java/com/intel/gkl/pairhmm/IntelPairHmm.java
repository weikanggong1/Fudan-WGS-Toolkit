package com.intel.gkl.pairhmm;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.nio.file.attribute.FileTime;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.broadinstitute.gatk.nativebindings.pairhmm.*;

/** Candidate dispatch adapter. The original GKL is never rewritten or scalarized.
 * Each batch stays intact; only its likelihood provider changes. Auto thresholds
 * are explicit calibration inputs, not a claim of an established speed benefit.
 */
public class IntelPairHmm implements PairHMMNativeBinding {
    boolean useOmp;
    boolean useFpga;
    void setNativeLibraryName(String ignored) { }
    private static final AtomicLong SEQUENCE = new AtomicLong();
    private static long activeCudaContext;
    private static boolean libraryLoaded;
    private static OriginalGklLoader originalLoader;
    private static String originalJar;
    private static String originalJarSha256;
    private static JarIdentity originalIdentity;
    private final long contextId = SEQUENCE.incrementAndGet();
    private final String mode = System.getProperty("fnit.pairhmm.mode", "");
    private long context;
    private long calls, cpuCalls, gpuCalls, cpuPairs, gpuPairs, cpuCells, gpuCells;
    private long minGpuPairs, minGpuCells;
    private PairHMMNativeBinding original;
    private PairHMMNativeArguments settings;
    private boolean loaded, initialized, finished;
    private long jarVerifyNanoseconds, originalConstructNanoseconds, originalLoadNanoseconds, originalInitializeNanoseconds;
    private long cudaLibraryLoadNanoseconds, cudaContextCreateNanoseconds;
    private JarIdentity verifiedIdentity;
    private String verificationMode="not_used", bindingSha256="";
    private static final long CAPTURE_BYTE_LIMIT=16L*1024*1024;
    private Path captureMaxPath;
    private CaptureBuffer largestCapture;
    private long observedMaxCells,observedMaxPairs,retainedCells,retainedPairs,captureReplacements,captureCloneNanoseconds,captureWriteNanoseconds;

    private static final class CaptureBuffer extends ByteArrayOutputStream {
        CaptureBuffer() { super((int)CAPTURE_BYTE_LIMIT); }
        @Override public synchronized void write(int value) {
            if(count==buf.length)throw new IllegalStateException("Capture buffer bound exceeded");
            super.write(value);
        }
        @Override public synchronized void write(byte[] bytes,int offset,int length) {
            if(length>buf.length-count)throw new IllegalStateException("Capture buffer bound exceeded");
            super.write(bytes,offset,length);
        }
    }

    /** Linux stat identity, with exact integer nanoseconds (no JSON doubles). */
    private static final class JarIdentity {
        final Path path;
        final long dev,ino,size,mtime,ctime;
        JarIdentity(Path path)throws IOException {
            this.path=path.toRealPath();
            Map<String,Object> a=Files.readAttributes(this.path,"unix:dev,ino,size,lastModifiedTime,ctime");
            dev=((Number)a.get("dev")).longValue();ino=((Number)a.get("ino")).longValue();size=((Number)a.get("size")).longValue();
            mtime=((FileTime)a.get("lastModifiedTime")).to(TimeUnit.NANOSECONDS);
            ctime=((FileTime)a.get("ctime")).to(TimeUnit.NANOSECONDS);
        }
        boolean same(JarIdentity other) {
            return path.equals(other.path) && dev==other.dev && ino==other.ino && size==other.size && mtime==other.mtime && ctime==other.ctime;
        }
        boolean matches(JsonObject o) {
            return path.toString().equals(o.get("canonical_path").getAsString()) && dev==o.get("dev").getAsLong() && ino==o.get("ino").getAsLong() && size==o.get("size").getAsLong() && mtime==o.get("mtime_ns").getAsLong() && ctime==o.get("ctime_ns").getAsLong();
        }
    }

    private static final class OriginalGklLoader extends URLClassLoader {
        OriginalGklLoader(URL jar, ClassLoader parent) { super(new URL[]{jar}, parent); }
        @Override protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            synchronized (getClassLoadingLock(name)) {
                Class<?> result = findLoadedClass(name);
                if (result == null) result = name.startsWith("com.intel.gkl.") ? findClass(name) : super.loadClass(name, false);
                if (resolve) resolveClass(result);
                return result;
            }
        }
    }

    private static String sha256(Path file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buffer = new byte[1024*1024];
            for (int n; (n=in.read(buffer))!=-1;) digest.update(buffer,0,n);
        }
        StringBuilder text = new StringBuilder();
        for (byte b : digest.digest()) text.append(String.format("%02x", b & 255));
        return text.toString();
    }

    private void authenticateOriginalJar(Path jar,String expected)throws Exception {
        long started=System.nanoTime();
        JarIdentity before=new JarIdentity(jar);
        String configured=System.getProperty("fnit.pairhmm.gatkJarBinding"), expectedBinding=System.getProperty("fnit.pairhmm.gatkJarBindingSha256");
        if(configured!=null || expectedBinding!=null) {
            if(configured==null || expectedBinding==null || !expectedBinding.matches("[0-9a-f]{64}") || !Paths.get(configured).isAbsolute()) throw new IllegalArgumentException("Binding path and SHA-256 must both be explicit");
            Path binding=Paths.get(configured);
            if(Files.size(binding)>16384)throw new IllegalArgumentException("Preauthenticated jar receipt bound");
            byte[] bytes=Files.readAllBytes(binding);MessageDigest digest=MessageDigest.getInstance("SHA-256");
            StringBuilder hash=new StringBuilder();for(byte b:digest.digest(bytes))hash.append(String.format("%02x",b & 255));
            if(!hash.toString().equals(expectedBinding))throw new IllegalStateException("Preauthenticated jar receipt SHA mismatch");
            JsonObject receipt=new JsonParser().parse(new String(bytes,StandardCharsets.UTF_8)).getAsJsonObject();
            if(!receipt.get("schema").getAsString().equals("fnit_actual_sha256_jar_identity_v1") || !receipt.get("sha256").getAsString().equals(expected) || !before.matches(receipt)) throw new IllegalStateException("Preauthenticated jar identity differs from actual file");
            verificationMode="preauthenticated_SHA256_receipt_and_exact_stat";bindingSha256=expectedBinding;
        } else {
            if(!sha256(jar).equals(expected))throw new IllegalStateException("Original GATK jar SHA-256 mismatch");
            verificationMode="whole_jar_SHA256";
        }
        if(!before.same(new JarIdentity(jar)))throw new IllegalStateException("Original jar changed during authentication");
        verifiedIdentity=before;jarVerifyNanoseconds=System.nanoTime()-started;
    }

    private void checkOriginalIdentity() {
        if(verifiedIdentity==null)return;
        try { if(!verifiedIdentity.same(new JarIdentity(verifiedIdentity.path)))throw new IllegalStateException("Original GATK jar identity changed after authentication"); }
        catch(IOException error) { throw new IllegalStateException("Cannot verify original GATK identity",error); }
    }

    private PairHMMNativeBinding originalBinding() {
        synchronized (IntelPairHmm.class) {
            try {
                String configured = System.getProperty("fnit.pairhmm.gatkJar");
                String expected = System.getProperty("fnit.pairhmm.gatkJarSha256");
                if (configured==null || expected==null || !expected.matches("[0-9a-f]{64}")) throw new IllegalStateException("Original GATK jar and SHA-256 are required");
                File jar = new File(configured).getCanonicalFile();
                if (!new File(configured).isAbsolute() || !jar.isFile()) throw new IllegalArgumentException("Original GATK jar must be absolute");
                if (originalLoader==null) {
                    authenticateOriginalJar(jar.toPath(),expected);
                    originalLoader=new OriginalGklLoader(jar.toURI().toURL(),PairHMMNativeBinding.class.getClassLoader());
                    originalJar=jar.toString(); originalJarSha256=expected;originalIdentity=verifiedIdentity;
                } else if (!originalJar.equals(jar.toString()) || !originalJarSha256.equals(expected)) throw new IllegalStateException("A different original GKL jar identity is already bound");
                else { verifiedIdentity=originalIdentity;verificationMode="same_JVM_previously_authenticated_identity";checkOriginalIdentity(); }
                String name=useOmp ? "com.intel.gkl.pairhmm.IntelPairHmmOMP" : "com.intel.gkl.pairhmm.IntelPairHmm";
                Class<?> implementation=originalLoader.loadClass(name);
                if (implementation.getClassLoader()!=originalLoader) throw new IllegalStateException("Original GKL child loader was bypassed");
                if (!new File(implementation.getProtectionDomain().getCodeSource().getLocation().toURI()).getCanonicalPath().equals(originalJar)) throw new IllegalStateException("Original GKL code source differs");
                return (PairHMMNativeBinding)implementation.newInstance();
            } catch (RuntimeException error) { throw error; }
            catch (Exception error) { throw new IllegalStateException("Cannot load original unmodified GKL", error); }
        }
    }

    @Override public synchronized boolean load(File temporaryDirectory) {
        if (loaded) return true;
        if (!mode.equals("auto") && !mode.equals("forceGPU")) throw new IllegalArgumentException("Explicit fnit.pairhmm.mode=auto or forceGPU is required");
        String captureConfigured=System.getProperty("fnit.pairhmm.captureMax");
        if(captureConfigured!=null) {
            captureMaxPath=Paths.get(captureConfigured);
            if(!captureMaxPath.isAbsolute() || Files.exists(captureMaxPath))throw new IllegalArgumentException("Maximum real batch capture requires a new absolute private path");
        }
        if (mode.equals("auto")) {
            minGpuPairs=Long.parseLong(System.getProperty("fnit.pairhmm.gpuMinPairs", "0"));
            minGpuCells=Long.parseLong(System.getProperty("fnit.pairhmm.gpuMinDpCells", "0"));
            if (minGpuPairs<1 || minGpuCells<1) throw new IllegalArgumentException("Auto requires positive explicit calibrated pair and DP-cell thresholds");
            long started=System.nanoTime(); original=originalBinding(); originalConstructNanoseconds=System.nanoTime()-started-jarVerifyNanoseconds;
            started=System.nanoTime();
            if (!original.load(temporaryDirectory)) throw new IllegalStateException("Original GKL native library did not load");
            originalLoadNanoseconds=System.nanoTime()-started;
            checkOriginalIdentity();
        }
        loaded=true;
        return true;
    }

    @Override public synchronized void initialize(PairHMMNativeArguments arguments) {
        if (!loaded || initialized || finished) throw new IllegalStateException("Invalid adapter lifecycle");
        if (useFpga) throw new IllegalArgumentException("FPGA is outside this dispatch contract");
        if (arguments==null || arguments.maxNumberOfThreads<1) throw new IllegalArgumentException("Native arguments required");
        if (mode.equals("forceGPU") && arguments.useDoublePrecision) throw new IllegalArgumentException("GPU candidate is FP32 only");
        settings=new PairHMMNativeArguments(); settings.maxNumberOfThreads=arguments.maxNumberOfThreads; settings.useDoublePrecision=arguments.useDoublePrecision;
        if (original!=null) { long started=System.nanoTime(); original.initialize(settings); originalInitializeNanoseconds=System.nanoTime()-started; }
        initialized=true;
        audit("initialized"); // Does not load libcudart or create a CUDA context.
    }

    private void ensureCuda() {
        if (context!=0) return;
        synchronized (IntelPairHmm.class) {
            if (activeCudaContext!=0) throw new IllegalStateException("Only one active project CUDA PairHMM context per JVM");
            activeCudaContext=contextId;
            try {
                if (!libraryLoaded) {
                    String configured=System.getProperty("fnit.pairhmm.library");
                    if (configured==null || !new File(configured).isAbsolute()) throw new IllegalArgumentException("Absolute project JNI library required");
                    File library=new File(configured).getCanonicalFile();
                    if (!library.isFile()) throw new IllegalStateException("Project JNI library missing");
                    long started=System.nanoTime(); System.load(library.toString()); cudaLibraryLoadNanoseconds=System.nanoTime()-started; libraryLoaded=true;
                }
                long started=System.nanoTime();
                context=createNative(Integer.parseInt(System.getProperty("fnit.pairhmm.device","0")),settings.maxNumberOfThreads);
                cudaContextCreateNanoseconds=System.nanoTime()-started;
                if (context==0) throw new IllegalStateException("CUDA context creation failed");
                audit("gpu_initialized");
            } catch (IOException error) { activeCudaContext=0; throw new IllegalStateException(error); }
            catch (RuntimeException | Error error) {
                if (context!=0) { try { destroyNative(context); } catch (RuntimeException | Error cleanup) { error.addSuppressed(cleanup); } }
                context=0; activeCudaContext=0; throw error;
            }
        }
    }

    @Override public synchronized void computeLikelihoods(ReadDataHolder[] reads,HaplotypeDataHolder[] haps,double[] output) {
        if (!initialized || finished) throw new IllegalStateException("Adapter not active");
        if (reads==null || haps==null || output==null || (long)reads.length*haps.length!=output.length) throw new IllegalArgumentException("Read-major output shape");
        long readLengths=0,hapLengths=0;
        for (ReadDataHolder r:reads) {
            if (r==null || r.readBases==null || r.readQuals==null || r.insertionGOP==null || r.deletionGOP==null || r.overallGCP==null) throw new IllegalArgumentException("Missing read array");
            int n=r.readBases.length;
            if(n<1 || r.readQuals.length!=n || r.insertionGOP.length!=n || r.deletionGOP.length!=n || r.overallGCP.length!=n) throw new IllegalArgumentException("Read array lengths");
            readLengths=Math.addExact(readLengths,n);
        }
        for(HaplotypeDataHolder h:haps) { if(h==null || h.haplotypeBases==null || h.haplotypeBases.length<1) throw new IllegalArgumentException("Empty haplotype"); hapLengths=Math.addExact(hapLengths,h.haplotypeBases.length); }
        long cells=Math.multiplyExact(readLengths,hapLengths), pairs=output.length;
        if(pairs==0) return;
        observedMaxCells=Math.max(observedMaxCells,cells);observedMaxPairs=Math.max(observedMaxPairs,pairs);
        captureMaximum(reads,haps,cells,pairs,readLengths,hapLengths);
        boolean gpu=mode.equals("forceGPU") || (!settings.useDoublePrecision && pairs>=minGpuPairs && cells>=minGpuCells);
        if(gpu) {
            ensureCuda();
            try { computeNative(context,reads,haps,output); }
            catch(RuntimeException error) { try { audit("gpu_failed"); } catch(RuntimeException observe) { error.addSuppressed(observe); } throw error; }
            ++gpuCalls; gpuPairs=Math.addExact(gpuPairs,pairs); gpuCells=Math.addExact(gpuCells,cells);
        } else {
            original.computeLikelihoods(reads,haps,output);
            ++cpuCalls; cpuPairs=Math.addExact(cpuPairs,pairs); cpuCells=Math.addExact(cpuCells,cells);
        }
        for(double v:output) if(!Double.isFinite(v) || v>1e-6) throw new IllegalStateException("Invalid provider log10 likelihood");
        ++calls;
        if((calls & (calls-1))==0) audit("computed");
    }

    @Override public synchronized void done() {
        if(!initialized || finished) return;
        RuntimeException problem=null;
        try {
            checkOriginalIdentity();
            if(largestCapture!=null) {
                long started=System.nanoTime();
                try(java.io.OutputStream output=Files.newOutputStream(captureMaxPath,StandardOpenOption.CREATE_NEW)) { largestCapture.writeTo(output); }
                catch(IOException error) { throw new IllegalStateException("Cannot write maximum actual batch capture",error); }
                finally { captureWriteNanoseconds=System.nanoTime()-started; }
            }
            audit("completed");
        } catch(RuntimeException error) { problem=error; }
        try { if(original!=null) original.done(); } catch(RuntimeException error) { if(problem==null)problem=error;else problem.addSuppressed(error); }
        try { if(context!=0)destroyNative(context); } catch(RuntimeException error) { if(problem==null)problem=error;else problem.addSuppressed(error); }
        finally { context=0;finished=true; synchronized(IntelPairHmm.class) { if(activeCudaContext==contextId)activeCudaContext=0; } }
        if(problem!=null)throw problem;
    }

    private void captureMaximum(ReadDataHolder[] reads,HaplotypeDataHolder[] haps,long cells,long pairs,long readLengths,long hapLengths) {
        if(captureMaxPath==null || (cells<retainedCells || (cells==retainedCells && pairs<=retainedPairs)))return;
        long bytes=Math.addExact(16L,Math.addExact(Math.multiplyExact(4L,(long)reads.length+haps.length),Math.addExact(Math.multiplyExact(5L,readLengths),hapLengths)));
        if(bytes>CAPTURE_BYTE_LIMIT || pairs>1000000L)return;
        long started=System.nanoTime();
        try {
            if(largestCapture==null)largestCapture=new CaptureBuffer();
            largestCapture.reset();
            try(DataOutputStream stream=new DataOutputStream(largestCapture)) {
                stream.writeLong(0x464e495450484d31L);stream.writeInt(reads.length);stream.writeInt(haps.length);
                for(ReadDataHolder r:reads) { stream.writeInt(r.readBases.length);stream.write(r.readBases);stream.write(r.readQuals);stream.write(r.insertionGOP);stream.write(r.deletionGOP);stream.write(r.overallGCP); }
                for(HaplotypeDataHolder h:haps) { stream.writeInt(h.haplotypeBases.length);stream.write(h.haplotypeBases); }
            }
            if(largestCapture.size()!=bytes)throw new IllegalStateException("Capture serialized size differs");
            retainedCells=cells;retainedPairs=pairs;++captureReplacements;
        } catch(IOException error) { throw new IllegalStateException("Cannot retain bounded actual batch",error); }
        finally { captureCloneNanoseconds+=System.nanoTime()-started; }
    }

    private void audit(String event) {
        String configured=System.getProperty("fnit.pairhmm.audit");
        if(configured==null || !Paths.get(configured).isAbsolute()) throw new IllegalArgumentException("Absolute private audit required");
        String json="{\"provider\":\"FNIT_ADAPTIVE_PAIRHMM_CANDIDATE\",\"context_id\":"+contextId+",\"event\":\""+event+"\",\"mode\":\""+mode+"\",\"gpu_initialized\":"+(context!=0)+",\"original_GKL_child_loader\":"+(original!=null)+",\"cpu_calls\":"+cpuCalls+",\"gpu_calls\":"+gpuCalls+",\"cpu_pairs\":"+cpuPairs+",\"gpu_pairs\":"+gpuPairs+",\"cpu_dp_cells\":"+cpuCells+",\"gpu_dp_cells\":"+gpuCells+",\"gpu_min_pairs\":"+minGpuPairs+",\"gpu_min_dp_cells\":"+minGpuCells+",\"original_jar_verification_mode\":\""+verificationMode+"\",\"original_jar_binding_sha256\":\""+bindingSha256+"\",\"setup_nanoseconds\":{\"original_jar_authentication\":"+jarVerifyNanoseconds+",\"original_class_construct\":"+originalConstructNanoseconds+",\"original_native_load\":"+originalLoadNanoseconds+",\"original_initialize\":"+originalInitializeNanoseconds+",\"CUDA_library_load\":"+cudaLibraryLoadNanoseconds+",\"CUDA_context_create\":"+cudaContextCreateNanoseconds+"},\"maximum_actual_batch\":{\"enabled\":"+(captureMaxPath!=null)+",\"byte_limit\":"+CAPTURE_BYTE_LIMIT+",\"observed_max_dp_cells\":"+observedMaxCells+",\"observed_max_pairs\":"+observedMaxPairs+",\"retained_dp_cells\":"+retainedCells+",\"retained_pairs\":"+retainedPairs+",\"retained_bytes\":"+(largestCapture==null?0:largestCapture.size())+",\"retained_replacements\":"+captureReplacements+",\"clone_nanoseconds\":"+captureCloneNanoseconds+",\"write_nanoseconds\":"+captureWriteNanoseconds+"},\"native\":"+(context==0?"null":statsNative(context))+"}\n";
        synchronized(IntelPairHmm.class) { try { Files.write(Paths.get(configured),json.getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE,StandardOpenOption.APPEND); } catch(IOException error) { throw new IllegalStateException("Cannot write adaptive provider audit",error); } }
    }

    private static native long createNative(int device,int maxThreads);
    private static native void computeNative(long context,ReadDataHolder[] reads,HaplotypeDataHolder[] haps,double[] output);
    private static native String statsNative(long context);
    private static native void destroyNative(long context);
}
