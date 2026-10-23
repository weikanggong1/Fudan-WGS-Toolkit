package org.fudan.wgs.gatkpatch;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Map;

/** Verify the classes in this scientific JVM before invoking only BQSR training. */
public final class SourcePatchedBqsrMain {
    private static final String ENGINE = "org.broadinstitute.hellbender.utils.recalibration.BaseRecalibrationEngine";
    private static final String INFO = "org.broadinstitute.hellbender.utils.recalibration.ReadRecalibrationInfo";
    private static final String SELF = "org.fudan.wgs.gatkpatch.SourcePatchedBqsrMain";
    private static final String PARALLEL = "org.fnit.wgs.bqsr.parallel.ParallelFusedBqsrTrainer";

    private static byte[] capture(InputStream input, int maximum) throws Exception {
        try (InputStream source = input; ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            if (source == null) throw new IllegalStateException("Missing bound class resource");
            byte[] buffer = new byte[4096];
            for (int n; (n = source.read(buffer)) != -1;) {
                if (output.size() + n > maximum) throw new IllegalStateException("Binding resource exceeds its byte allowance");
                output.write(buffer, 0, n);
            }
            return output.toByteArray();
        }
    }
    private static String digest(byte[] data) throws Exception {
        StringBuilder result = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(data)) result.append(String.format("%02x", value & 255));
        return result.toString();
    }
    private static Path origin(Class<?> type) throws Exception {
        if (type.getProtectionDomain().getCodeSource() == null) throw new IllegalStateException("Class CodeSource is unavailable: " + type.getName());
        return Paths.get(type.getProtectionDomain().getCodeSource().getLocation().toURI()).toRealPath();
    }
    private static JsonObject verify(String mode) throws Exception {
        Path path = Paths.get(System.getProperty("fudan.gatkpatch.binding"));
        byte[] data = capture(Files.newInputStream(path), 131072);
        if (!digest(data).equals(System.getProperty("fudan.gatkpatch.bindingSha256"))) throw new IllegalStateException("Class binding manifest SHA256 differs");
        JsonObject binding = new JsonParser().parse(new String(data, StandardCharsets.UTF_8)).getAsJsonObject();
        if (!binding.get("schema").getAsString().equals("fudan-original-gatk-sourcepatch-classes-v1")) throw new IllegalStateException("Unknown class binding schema");
        Path overlay = Paths.get(binding.get("sourcePatchJar").getAsString()).toRealPath();
        JsonObject expected = binding.getAsJsonObject("classes"), loaded = new JsonObject();
        if (!expected.has(ENGINE) || !expected.has(INFO) || !expected.has(SELF)) throw new IllegalStateException("Incomplete source-patch class manifest");
        ClassLoader loader = SourcePatchedBqsrMain.class.getClassLoader();
        for (Map.Entry<String, JsonElement> item : expected.entrySet()) {
            String name = item.getKey();
            if (!(name.equals(ENGINE) || name.startsWith(ENGINE + "$") || name.equals(INFO)
                    || name.startsWith(INFO + "$") || name.equals(SELF) || name.startsWith(SELF + "$"))) throw new IllegalStateException("Unexpected overlay class: " + name);
            Class<?> type = Class.forName(name, false, loader);
            if (!origin(type).equals(overlay)) throw new IllegalStateException("Training class is not loaded from the bound source patch: " + name);
            String sha = digest(capture(type.getResourceAsStream("/" + name.replace('.', '/') + ".class"), 1048576));
            if (!sha.equals(item.getValue().getAsString())) throw new IllegalStateException("Actual loaded class resource SHA differs: " + name);
            JsonObject record = new JsonObject(); record.addProperty("code_source", origin(type).toString()); record.addProperty("sha256", sha); loaded.add(name, record);
        }
        Class<?> main = Class.forName("org.broadinstitute.hellbender.Main", false, loader);
        if (!origin(main).equals(Paths.get(binding.get("gatkJar").getAsString()).toRealPath())) throw new IllegalStateException("Original GATK Main jar differs");
        String mainSha = digest(capture(main.getResourceAsStream("/org/broadinstitute/hellbender/Main.class"), 1048576));
        if (!mainSha.equals(binding.get("original_main_class_sha256").getAsString())) throw new IllegalStateException("Original GATK Main resource SHA differs");
        JsonObject audit = new JsonObject(); audit.addProperty("schema", "fudan-original-gatk-sourcepatch-runtime-v1");
        audit.addProperty("status", "classes_verified_before_training"); audit.addProperty("entrypoint", mode);
        audit.addProperty("binding_sha256", digest(data)); audit.add("classes", loaded);
        audit.addProperty("original_main_code_source", origin(main).toString()); audit.addProperty("original_main_class_sha256", mainSha);
        return audit;
    }
    public static void main(String[] args) throws Exception {
        if (args.length < 2 || !(args[0].equals("probe") || args[0].equals("original") || args[0].equals("parallel"))) throw new IllegalArgumentException("Explicit probe/original/parallel mode and new private audit path required");
        String mode = args[0]; String[] forwarded = Arrays.copyOfRange(args, 2, args.length);
        if ((mode.equals("probe") && forwarded.length != 0) || (mode.equals("original") && (forwarded.length == 0 || !forwarded[0].equals("BaseRecalibrator")))) throw new IllegalArgumentException("This source patch is restricted to original BQSR training");
        JsonObject audit = verify(mode);
        Files.write(Paths.get(args[1]), (new GsonBuilder().setPrettyPrinting().create().toJson(audit) + "\n").getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        if (mode.equals("probe")) return;
        try {
            Class<?> main = Class.forName(mode.equals("original") ? "org.broadinstitute.hellbender.Main" : PARALLEL);
            main.getMethod("main", String[].class).invoke(null, (Object)forwarded);
        } catch (InvocationTargetException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof Error) throw (Error)cause;
            if (cause instanceof Exception) throw (Exception)cause;
            throw new IllegalStateException("Original BQSR entrypoint failed", cause);
        }
    }
}
