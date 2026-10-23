import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import com.google.gson.*;
import org.broadinstitute.hellbender.tools.walkers.haplotypecaller.HaplotypeCallerEngine;
/** Build-only source/version probe. Does not read any genomic input. */
public final class HcGuardProbe {
 public static void main(String[] args)throws Exception {
  HCMappedStartGuard.need(args.length==3,"jar expected_sha new_output");
  Path jar=Paths.get(args[0]).toRealPath(),out=Paths.get(args[2]);
  HCMappedStartGuard.need(!Files.exists(out),"new probe output required");
  String sha=HCMappedStartGuard.sha(jar);
  HCMappedStartGuard.need(sha.equals(args[1])&&sha.matches("[0-9a-f]{64}"),"original jar SHA mismatch");
  Path origin=Paths.get(HaplotypeCallerEngine.class.getProtectionDomain().getCodeSource().getLocation().toURI()).toRealPath();
  HCMappedStartGuard.need(origin.equals(jar),"original HCE source mismatch");
  String version=HaplotypeCallerEngine.class.getPackage().getImplementationVersion();
  HCMappedStartGuard.need("4.2.0.0".equals(version),"original GATK version mismatch");
  JsonObject p=new JsonObject();p.addProperty("status","original_GATK_guard_class_origin_version_probe_complete_not_science");p.addProperty("original_GATK_version",version);p.addProperty("original_gatk_jar_sha256",sha);p.addProperty("original_HCE_code_source",origin.toString());
  Files.write(out,(new GsonBuilder().setPrettyPrinting().create().toJson(p)+"\n").getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE_NEW);
 }
}
