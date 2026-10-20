package org.fnit.wgs;

import com.intel.gkl.pairhmm.IntelPairHmmOMP;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import org.broadinstitute.gatk.nativebindings.pairhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pairhmm.ReadDataHolder;

/** Same bounded private batch against either the unmodified GKL or project JNI.
 * CPU: probe.jar:gatk.jar. GPU: adapter.jar:probe.jar:gatk.jar.
 */
public final class PairHmmBatchProbe {
    private static final long MAGIC = 0x464E495450484D31L;
    private static byte[] array(DataInputStream stream, int size) throws Exception {
        if (size < 1 || size > 1000000) throw new IllegalArgumentException("Invalid bounded fixture length");
        byte[] result = new byte[size]; stream.readFully(result); return result;
    }
    public static void main(String[] arguments) throws Exception {
        if (arguments.length == 1 && arguments[0].equals("--class-only")) {
            if (!PairHMMNativeBinding.class.isAssignableFrom(IntelPairHmmOMP.class)) throw new IllegalStateException("Binding type mismatch");
            new IntelPairHmmOMP();
            boolean actualAdapter;
            try { IntelPairHmmOMP.class.getSuperclass().getDeclaredMethod("createNative", int.class, int.class); actualAdapter = true; }
            catch (NoSuchMethodException missing) { actualAdapter = false; }
            if (Boolean.parseBoolean(System.getProperty("fnit.pairhmm.expectAdapter", "false")) != actualAdapter) throw new IllegalStateException("Unexpected classpath provider origin");
            System.out.println("{\"status\":\"classpath_contract_passed\",\"scientific_validation\":false,\"provider_class\":\"" + IntelPairHmmOMP.class.getName() + "\"}");
            return;
        }
        if (arguments.length != 2) throw new IllegalArgumentException("Usage: PairHmmBatchProbe PRIVATE.batch NEW.likelihoods.bin; or --class-only");
        final Path fixture = Paths.get(arguments[0]); final Path output = Paths.get(arguments[1]);
        if (Files.size(fixture) > 16L * 1024 * 1024) throw new IllegalArgumentException("Fixture exceeds 16 MiB bound");
        ReadDataHolder[] reads; HaplotypeDataHolder[] haps;
        try (DataInputStream stream = new DataInputStream(Files.newInputStream(fixture))) {
            if (stream.readLong() != MAGIC) throw new IllegalArgumentException("Wrong fixture format");
            int nreads = stream.readInt(), nhaps = stream.readInt();
            if (nreads < 1 || nhaps < 1 || (long)nreads * nhaps > 1000000L) throw new IllegalArgumentException("Fixture pair bound exceeded");
            reads = new ReadDataHolder[nreads]; haps = new HaplotypeDataHolder[nhaps];
            for (int i = 0; i < nreads; ++i) {
                int size = stream.readInt(); ReadDataHolder read = new ReadDataHolder();
                read.readBases = array(stream,size); read.readQuals = array(stream,size);
                read.insertionGOP = array(stream,size); read.deletionGOP = array(stream,size); read.overallGCP = array(stream,size); reads[i] = read;
            }
            for (int i = 0; i < nhaps; ++i) { HaplotypeDataHolder hap = new HaplotypeDataHolder(); hap.haplotypeBases = array(stream,stream.readInt()); haps[i] = hap; }
            if (stream.read() != -1) throw new IllegalArgumentException("Trailing private fixture bytes");
        }
        PairHMMNativeBinding binding = new IntelPairHmmOMP();
        if (!binding.load(null)) throw new IllegalStateException("Selected native provider failed to load");
        PairHMMNativeArguments settings = new PairHMMNativeArguments(); settings.maxNumberOfThreads = 1; settings.useDoublePrecision = false;
        double[] likelihoods = new double[reads.length * haps.length]; long started = System.nanoTime();
        binding.initialize(settings);
        try { binding.computeLikelihoods(reads, haps, likelihoods); }
        finally { binding.done(); }
        long nanoseconds = System.nanoTime() - started;
        for (double value : likelihoods) if (!Double.isFinite(value)) throw new IllegalStateException("Non-finite likelihood");
        try (DataOutputStream stream = new DataOutputStream(Files.newOutputStream(output, StandardOpenOption.CREATE_NEW))) {
            stream.writeLong(MAGIC); stream.writeInt(reads.length); stream.writeInt(haps.length);
            for (double value : likelihoods) stream.writeDouble(value);
        }
        System.out.println("{\"status\":\"batch_computed_requires_independent_comparison\",\"reads\":" + reads.length + ",\"haplotypes\":" + haps.length + ",\"pairs\":" + likelihoods.length + ",\"wall_nanoseconds\":" + nanoseconds + "}");
    }
}
