import htsjdk.samtools.*;
import htsjdk.samtools.util.BlockCompressedInputStream;
import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.FileTime;
import java.security.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.broadinstitute.hellbender.engine.filters.ReadFilter;
import org.broadinstitute.hellbender.engine.filters.ReadFilterLibrary;
import org.broadinstitute.hellbender.engine.spark.AssemblyRegionArgumentCollection;
import org.broadinstitute.hellbender.tools.walkers.haplotypecaller.HaplotypeCallerEngine;
import org.broadinstitute.hellbender.transformers.IUPACReadTransformer;
import org.broadinstitute.hellbender.transformers.ReadTransformer;
import org.broadinstitute.hellbender.utils.read.GATKRead;
import org.broadinstitute.hellbender.utils.read.SAMRecordToGATKReadAdapter;

/** DRAFT: one original-HTSJDK sequential BAM traversal; no reads are rewritten. */
public final class HCMappedStartGuard {
    private static final int LIMIT=50;
    static void need(boolean v,String reason){if(!v)throw new IllegalArgumentException(reason);}
    static String hex(byte[] b){StringBuilder s=new StringBuilder();for(byte x:b)s.append(String.format("%02x",x&255));return s.toString();}
    static String sha(Path p)throws Exception{MessageDigest d=MessageDigest.getInstance("SHA-256");try(InputStream s=Files.newInputStream(p)){byte[] b=new byte[1<<20];int n;while((n=s.read(b))!=-1)d.update(b,0,n);}return hex(d.digest());}
    static JsonObject identity(Path p)throws Exception{
        p=p.toRealPath();Map<String,Object>a=Files.readAttributes(p,"unix:dev,ino,size,lastModifiedTime,ctime");JsonObject o=new JsonObject();o.addProperty("path",p.toString());o.addProperty("device",((Number)a.get("dev")).longValue());o.addProperty("inode",((Number)a.get("ino")).longValue());o.addProperty("size",((Number)a.get("size")).longValue());o.addProperty("mtime_ns",((FileTime)a.get("lastModifiedTime")).to(TimeUnit.NANOSECONDS));o.addProperty("ctime_ns",((FileTime)a.get("ctime")).to(TimeUnit.NANOSECONDS));return o;
    }
    static final class CountedSHA extends InputStream {
        final InputStream input;final MessageDigest digest;long bytes;
        CountedSHA(InputStream s)throws Exception{input=s;digest=MessageDigest.getInstance("SHA-256");}
        public int read()throws IOException{int v=input.read();if(v!=-1){digest.update((byte)v);bytes++;}return v;}
        public int read(byte[] b,int off,int len)throws IOException{int n=input.read(b,off,len);if(n>0){digest.update(b,off,n);bytes+=n;}return n;}
        public long skip(long n)throws IOException{byte[] b=new byte[65536];long total=0;while(total<n){int k=read(b,0,(int)Math.min(b.length,n-total));if(k<0)break;total+=k;}return total;}
        public void close(){} // SamReader owns a non-closing view; outer raw stream closes once.
    }
    static final class Counts {
        long mapped,nonDuplicate,maximumMapped,maximumNonDuplicate,rawRiskStarts,nonDuplicateRiskStarts,filterPassing,maximumFilterPassing,filterPassingRiskStarts;
        JsonObject json(String name,int length){JsonObject o=new JsonObject();o.addProperty("contig",name);o.addProperty("length",length);o.addProperty("mapped_all_including_duplicates",mapped);o.addProperty("mapped_nonduplicate_upper_bound",nonDuplicate);o.addProperty("max_mapped_at_alignment_start",maximumMapped);o.addProperty("max_nonduplicate_at_alignment_start",maximumNonDuplicate);o.addProperty("raw_starts_over_50",rawRiskStarts);o.addProperty("nonduplicate_starts_over_50",nonDuplicateRiskStarts);o.addProperty("exact_original_filter_passing",filterPassing);o.addProperty("max_exact_filter_passing_at_alignment_start",maximumFilterPassing);o.addProperty("exact_filter_passing_starts_over_50",filterPassingRiskStarts);return o;}
        void group(long all,long nonDup,long passed){maximumFilterPassing=Math.max(maximumFilterPassing,passed);if(passed>LIMIT)filterPassingRiskStarts++;maximumMapped=Math.max(maximumMapped,all);maximumNonDuplicate=Math.max(maximumNonDuplicate,nonDup);if(all>LIMIT)rawRiskStarts++;if(nonDup>LIMIT)nonDuplicateRiskStarts++;}
    }
    static JsonObject proveDefaultUpperBound(Path jar,String expectedSHA,JsonArray command,Path bam,Path fai)throws Exception{
        need(expectedSHA.matches("[0-9a-f]{64}")&&sha(jar).equals(expectedSHA),"original GATK jar SHA mismatch");
        Path origin=Paths.get(HaplotypeCallerEngine.class.getProtectionDomain().getCodeSource().getLocation().toURI()).toRealPath();need(origin.equals(jar.toRealPath()),"guard loaded an overlay HaplotypeCallerEngine");
        List<String> args=new ArrayList<>();for(JsonElement e:command)args.add(e.getAsString());need(Collections.frequency(args,"HaplotypeCaller")==1,"one exact original HC command required");
        boolean gvcf=false;Set<String> forbidden=new HashSet<>(Arrays.asList("--disable-tool-default-read-filters","--disable-read-filter","-DF","--read-filter","-RF","--gatk-config-file","--dragen-mode","--transform-dragen-mapping-quality","--annotation","-A","--annotation-group","-G","--disable-tool-default-annotations"));
        for(int i=0;i<args.size();i++){
            String option=args.get(i).split("=",2)[0];need(!forbidden.contains(option),"nondefault filtering/transform argument forbidden");
            if(option.equals("--emit-ref-confidence")||option.equals("-ERC")){need(i+1<args.size(),"missing GVCF argument");gvcf=args.get(++i).equals("GVCF");}
            if(option.equals("--max-reads-per-alignment-start")){need(i+1<args.size()&&Integer.parseInt(args.get(++i))==LIMIT,"only original cap50 in this draft");}
        }
        int input=args.indexOf("-I"),reference=args.indexOf("-R");
        need(input>=0&&input+1<args.size()&&Collections.frequency(args,"-I")==1&&Paths.get(args.get(input+1)).toRealPath().equals(bam),"one exact scanned BAM must be the HC input");
        need(reference>=0&&reference+1<args.size()&&Paths.get(args.get(reference+1)+".fai").toRealPath().equals(fai),"HC reference FAI must equal scanned dictionary");
        need(gvcf,"fixed HC GVCF command required");need(new AssemblyRegionArgumentCollection().maxReadsPerAlignmentStart==LIMIT,"original default cap is not50");
        List<ReadFilter> filters=HaplotypeCallerEngine.makeStandardHCReadFilters();need(filters.contains(ReadFilterLibrary.MAPPED)&&filters.contains(ReadFilterLibrary.NOT_DUPLICATE),"original default MAPPED/NOT_DUPLICATE filters absent");
        need(HaplotypeCallerEngine.makeStandardHCReadTransformer().getClass()==IUPACReadTransformer.class,"default pretransform is not proven IUPAC-only");
        JsonObject p=new JsonObject();p.addProperty("enabled",true);p.addProperty("proof","all HC-passing reads are mapped and not duplicate; IUPAC pretransform changes bases only; all other default filters can only reduce this upper bound");p.addProperty("original_gatk_jar_sha256",expectedSHA);JsonArray names=new JsonArray();for(ReadFilter f:filters)names.add(f.getClass().getName());p.add("actual_default_filter_classes",names);p.addProperty("per_read_default_filter_chain_reimplemented",false);p.addProperty("scope","fixed original GATK4.2 HC GVCF command only; not generic HC/GenotypeGVCFs or other read transforms");return p;
    }
    public static void main(String[] argv)throws Exception{
        need(argv.length==6,"bam fai original_gatk.jar expected_jar_sha hc_command.json new_guard.json");Path bam=Paths.get(argv[0]).toRealPath(),fai=Paths.get(argv[1]).toRealPath(),jar=Paths.get(argv[2]).toRealPath(),commandPath=Paths.get(argv[4]).toRealPath(),out=Paths.get(argv[5]);need(!Files.exists(out),"new output required");
        long began=System.nanoTime();JsonObject bamBefore=identity(bam),faiBefore=identity(fai),commandBefore=identity(commandPath),jarBefore=identity(jar);need(Files.size(fai)<8L<<20&&Files.size(commandPath)<256L<<10,"bounded metadata only");
        JsonArray command=new JsonParser().parse(new String(Files.readAllBytes(commandPath),StandardCharsets.UTF_8)).getAsJsonArray();JsonObject proof=proveDefaultUpperBound(jar,argv[3],command,bam,fai);
        List<String>names=new ArrayList<>();List<Integer>lengths=new ArrayList<>();Set<String>unique=new HashSet<>();
        for(String line:Files.readAllLines(fai,StandardCharsets.UTF_8)){String[] a=line.split("\t");need(a.length==5&&unique.add(a[0]),"invalid/duplicate FAI sequence");long n=Long.parseLong(a[1]);need(n>0&&n<=Integer.MAX_VALUE,"unsupported sequence length");names.add(a[0]);lengths.add((int)n);}need(!names.isEmpty(),"empty reference dictionary");
        need(BlockCompressedInputStream.checkTermination(bam.toFile())==BlockCompressedInputStream.FileTermination.HAS_TERMINATOR_BLOCK,"BAM EOF terminator missing");Counts[] counts=new Counts[names.size()];for(int i=0;i<counts.length;i++)counts[i]=new Counts();
        long records=0,mapped=0,unmapped=0,groupAll=0,groupNonDup=0,groupPassed=0,scanBegan=System.nanoTime();int groupRef=-1,groupPos=-1;String bamSHA;long compressedBytes;
        try(InputStream raw=Files.newInputStream(bam)){
            CountedSHA stream=new CountedSHA(raw);SamReaderFactory factory=SamReaderFactory.makeDefault().validationStringency(ValidationStringency.STRICT).enable(SamReaderFactory.Option.VALIDATE_CRC_CHECKSUMS);
            try(SamReader reader=factory.open(SamInputResource.of(stream))){
                need(reader.type()==SamReader.Type.BAM_TYPE,"guard requires BAM, not SAM/CRAM");SAMFileHeader header=reader.getFileHeader();need(header.getSortOrder()==SAMFileHeader.SortOrder.coordinate,"BAM header not coordinate sorted");List<SAMSequenceRecord> seq=header.getSequenceDictionary().getSequences();need(seq.size()==names.size(),"reference dictionary count differs");for(int i=0;i<seq.size();i++)need(seq.get(i).getSequenceName().equals(names.get(i))&&seq.get(i).getSequenceLength()==lengths.get(i),"ordered BAM/FAI dictionary differs");
                SAMRecord previous=null;SAMRecordCoordinateComparator order=new SAMRecordCoordinateComparator();ReadFilter originalFilter=ReadFilter.fromList(HaplotypeCallerEngine.makeStandardHCReadFilters(),header);ReadTransformer originalPre=HaplotypeCallerEngine.makeStandardHCReadTransformer();
                try(SAMRecordIterator reads=reader.iterator()){
                    while(reads.hasNext()){
                        SAMRecord r=reads.next();records++;if(previous!=null)need(order.fileOrderCompare(previous,r)<=0,"BAM record coordinate order decreases");previous=r;
                        int initialRef=r.getReferenceIndex(),initialPos=r.getAlignmentStart();boolean initialMapped=!r.getReadUnmappedFlag(),initialDuplicate=r.getDuplicateReadFlag();
                        GATKRead processed=originalPre.apply(new SAMRecordToGATKReadAdapter(r));boolean passed=originalFilter.test(processed);
                        need(r.getReferenceIndex()==initialRef&&r.getAlignmentStart()==initialPos&&(!r.getReadUnmappedFlag())==initialMapped&&r.getDuplicateReadFlag()==initialDuplicate,"original pretransform changed coordinate/flags");
                        if(!initialMapped){need(!passed,"original mapped filter accepted unmapped read");unmapped++;continue;}
                        int ref=initialRef,pos=initialPos;need(ref>=0&&ref<counts.length&&pos>=1&&pos<=lengths.get(ref),"mapped coordinate outside dictionary");
                        if(ref!=groupRef||pos!=groupPos){if(groupRef>=0)counts[groupRef].group(groupAll,groupNonDup,groupPassed);groupRef=ref;groupPos=pos;groupAll=groupNonDup=groupPassed=0;}
                        mapped++;groupAll++;counts[ref].mapped++;if(passed){groupPassed++;counts[ref].filterPassing++;}if(!initialDuplicate){groupNonDup++;counts[ref].nonDuplicate++;}
                    }
                }
            }
            if(groupRef>=0)counts[groupRef].group(groupAll,groupNonDup,groupPassed);byte[] remaining=new byte[65536];while(stream.read(remaining)!=-1){}compressedBytes=stream.bytes;bamSHA=hex(stream.digest.digest());
        }
        double scanSeconds=(System.nanoTime()-scanBegan)*1e-9;String faiSHA=sha(fai),commandSHA=sha(commandPath);need(identity(bam).equals(bamBefore)&&identity(fai).equals(faiBefore)&&identity(commandPath).equals(commandBefore)&&identity(jar).equals(jarBefore),"input identity changed during complete scan");need(compressedBytes==bamBefore.get("size").getAsLong(),"compressed SHA did not cover all BAM bytes");
        JsonObject result=new JsonObject();result.addProperty("status","complete_original_HTSJDK_guard_not_parallel_scientific_equivalence");result.addProperty("actual_exit",0);result.add("bam_identity",bamBefore);result.addProperty("full_compressed_bam_sha256",bamSHA);result.addProperty("full_bam_compressed_bytes_hashed",compressedBytes);result.addProperty("fai_sha256",faiSHA);result.addProperty("command_sha256",commandSHA);result.add("mapped_nonduplicate_bound_proof",proof);result.addProperty("raw_records",records);result.addProperty("mapped_records_including_duplicates",mapped);result.addProperty("unmapped_records",unmapped);result.addProperty("scan_seconds",scanSeconds);result.addProperty("whole_seconds",(System.nanoTime()-began)*1e-9);result.addProperty("all_HTSJDK_visible_records_scanned_once",true);result.addProperty("HTSJDK_BGZF_CRC_validation_enabled",true);result.addProperty("BGZF_final_terminator_checked",true);result.addProperty("does_not_prove_all_RNG_calls_absent",true);result.addProperty("exact_filter_chain_scope","original GATK default HC filters + IUPAC pretransform; fixed command only, posttransform identity; count only, no analysis");
        JsonArray rows=new JsonArray();for(int i=0;i<counts.length;i++)rows.add(counts[i].json(names.get(i),lengths.get(i)));result.add("per_contig",rows);Files.write(out,(new GsonBuilder().setPrettyPrinting().create().toJson(result)+"\n").getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE_NEW);System.out.println("complete mapped-start guard; private counts saved; parallel equivalence not assessed");
    }
}
