package org.fnit.wgs;

import com.google.gson.*;
import htsjdk.samtools.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import org.broadinstitute.hellbender.utils.report.GATKReport;
import org.broadinstitute.hellbender.utils.report.GATKReportTable;

/** Streaming complete-field comparisons with the actual GATK jar's HTSJDK. */
public final class BqsrComparator {
    private static final String[] CORE={"QNAME","FLAG","RNAME","POS","MAPQ","CIGAR","RNEXT","PNEXT","TLEN","SEQ","QUAL"};
    private static void add(Map<String,Long> counts,String name) { counts.put(name,counts.getOrDefault(name,0L)+1); }
    private static String hash(String value)throws Exception {
        byte[] digest=MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
        StringBuilder s=new StringBuilder();for(byte b:digest)s.append(String.format("%02x",b&255));return s.toString();
    }
    private static String header(SAMFileHeader h) { StringWriter out=new StringWriter();new SAMTextHeaderCodec().encode(out,h);return out.toString(); }
    private static Map<String,String> tags(String[] fields) {
        Map<String,String> result=new TreeMap<>();
        for(int i=11;i<fields.length;i++) {
            if(fields[i].length()<5 || fields[i].charAt(2)!=':' || fields[i].charAt(4)!=':')throw new IllegalStateException("Invalid typed AUX");
            if(result.put(fields[i].substring(0,2),fields[i].substring(3))!=null)throw new IllegalStateException("Duplicate AUX tag");
        }
        return result;
    }
    private static Map<String,Map<String,String>> headerRows(String text) {
        Map<String,Map<String,String>> result=new LinkedHashMap<>();Map<String,Integer> ordinals=new HashMap<>();
        for(String line:text.split("\\r?\\n")) {
            if(line.isEmpty())continue;String[] f=line.split("\\t",-1);String kind=f[0].substring(1);
            Map<String,String> fields=new TreeMap<>();
            for(int i=1;i<f.length;i++) {int p=f[i].indexOf(':');fields.put(p<0?"text"+i:f[i].substring(0,p),p<0?f[i]:f[i].substring(p+1));}
            String entity=kind.equals("SQ")?fields.get("SN"):kind.equals("RG")||kind.equals("PG")?fields.get("ID"):null;
            if(entity==null) {int n=ordinals.getOrDefault(kind,0);ordinals.put(kind,n+1);entity=Integer.toString(n);}
            if(result.put(kind+":"+entity,fields)!=null)throw new IllegalStateException("Duplicate header identity");
        }
        return result;
    }
    private static String normalize(String text,Map<String,String> roles) {
        List<String> keys=new ArrayList<>(roles.keySet());keys.sort((a,b)->Integer.compare(b.length(),a.length()));
        for(String key:keys)text=text.replace(key,roles.get(key));return text;
    }
    private static boolean referenceAnnotations(SAMFileHeader h,Path reference,Path dictionary)throws Exception {
        Map<String,String> md5=new HashMap<>();
        for(String line:Files.readAllLines(dictionary,StandardCharsets.UTF_8))if(line.startsWith("@SQ\t")) {
            String name=null,value=null;for(String field:line.split("\t")) {if(field.startsWith("SN:"))name=field.substring(3);if(field.startsWith("M5:"))value=field.substring(3);}
            if(name!=null)md5.put(name,value);
        }
        for(SAMSequenceRecord s:h.getSequenceDictionary().getSequences()) {
            String value=s.getAttribute("M5");if(value!=null&&!value.equals(md5.get(s.getSequenceName())))return false;
            value=s.getAttribute("UR");
            if(value!=null) {Path uri=value.startsWith("file:")?Paths.get(new java.net.URI(value)):Paths.get(value);if(!uri.toRealPath().equals(reference.toRealPath()))return false;}
        }
        return true;
    }
    private static JsonObject align(String[] args)throws Exception {
        Path reference=Paths.get(args[1]),dictionary=Paths.get(args[2]);
        JsonObject roleJson=new JsonParser().parse(new String(Files.readAllBytes(Paths.get(args[5])),StandardCharsets.UTF_8)).getAsJsonObject();
        Map<String,String> roles=new HashMap<>();for(Map.Entry<String,JsonElement> e:roleJson.entrySet())roles.put(e.getKey(),e.getValue().getAsString());
        SamReaderFactory factory=SamReaderFactory.makeDefault().validationStringency(ValidationStringency.SILENT).referenceSequence(reference.toFile());
        JsonObject report=new JsonObject();Map<String,Long> coreDiff=new TreeMap<>(),auxDiff=new TreeMap<>(),auxKinds=new TreeMap<>(),headerDiff=new TreeMap<>();
        long leftCount=0,rightCount=0,different=0,rawDifferent=0;
        try(SamReader a=factory.open(new File(args[3]));SamReader b=factory.open(new File(args[4]));SAMRecordIterator ia=a.iterator();SAMRecordIterator ib=b.iterator()) {
            SAMFileHeader ah=a.getFileHeader(),bh=b.getFileHeader();String at=header(ah),bt=header(bh);
            Map<String,Map<String,String>> ar=headerRows(at),br=headerRows(bt);Set<String> entities=new LinkedHashSet<>(ar.keySet());entities.addAll(br.keySet());
            boolean normalized=true,rg=true;
            for(String entity:entities) {
                Map<String,String> x=ar.get(entity),y=br.get(entity);String kind=entity.substring(0,entity.indexOf(':'));
                if(x==null||y==null) {add(headerDiff,kind+":record_presence");normalized=false;if(kind.equals("RG"))rg=false;continue;}
                Set<String> keys=new TreeSet<>(x.keySet());keys.addAll(y.keySet());
                for(String key:keys)if(!Objects.equals(x.get(key),y.get(key))) {
                    add(headerDiff,kind+":"+key);if(kind.equals("RG"))rg=false;
                    if(kind.equals("SQ")&&(key.equals("M5")||key.equals("UR")))continue;
                    if(kind.equals("PG")&&key.equals("CL")&&normalize(x.getOrDefault(key,""),roles).equals(normalize(y.getOrDefault(key,""),roles)))continue;
                    normalized=false;
                }
            }
            List<SAMSequenceRecord> as=ah.getSequenceDictionary().getSequences(),bs=bh.getSequenceDictionary().getSequences();boolean sq=as.size()==bs.size();
            if(sq)for(int i=0;i<as.size();i++)sq&=as.get(i).getSequenceName().equals(bs.get(i).getSequenceName())&&as.get(i).getSequenceLength()==bs.get(i).getSequenceLength();
            boolean annotations=referenceAnnotations(ah,reference,dictionary)&&referenceAnnotations(bh,reference,dictionary);
            report.addProperty("header_text_equal",at.equals(bt));report.addProperty("header_left_sha256",hash(at));report.addProperty("header_right_sha256",hash(bt));
            report.addProperty("sequence_names_lengths_order_equal",sq);report.addProperty("read_groups_all_fields_equal",rg);
            report.addProperty("present_M5_and_UR_match_bound_reference",annotations);
            report.addProperty("header_equal_after_explicit_path_roles_and_verified_reference_annotation",normalized&&sq&&rg&&annotations);
            report.add("header_different_fields",new Gson().toJsonTree(headerDiff));
            while(ia.hasNext()||ib.hasNext()) {
                SAMRecord x=ia.hasNext()?ia.next():null,y=ib.hasNext()?ib.next():null;
                if(x!=null)leftCount++;if(y!=null)rightCount++;
                if(x==null||y==null){different++;rawDifferent++;continue;}
                String xs=x.getSAMString().replaceFirst("[\\r\\n]+$",""),ys=y.getSAMString().replaceFirst("[\\r\\n]+$","");
                if(!xs.equals(ys))rawDifferent++;
                String[] xf=xs.split("\\t",-1),yf=ys.split("\\t",-1);if(xf.length<11||yf.length<11)throw new IllegalStateException("Short SAM row");
                boolean same=true;for(int i=0;i<11;i++)if(!xf[i].equals(yf[i])){add(coreDiff,CORE[i]);same=false;}
                Map<String,String> xt=tags(xf),yt=tags(yf);Set<String> keys=new TreeSet<>(xt.keySet());keys.addAll(yt.keySet());
                for(String key:keys)if(!Objects.equals(xt.get(key),yt.get(key))){
                    add(auxDiff,key);same=false;
                    add(auxKinds,key+":"+(xt.get(key)==null?"right_only":yt.get(key)==null?"left_only":"both_present_different"));
                }
                if(!same)different++;
            }
        }
        report.addProperty("status","decoded_complete_records");report.addProperty("decoder","same_GATK_jar_HTSJDK");
        report.addProperty("records_left",leftCount);report.addProperty("records_right",rightCount);report.addProperty("different_complete_record_rows",different);report.addProperty("different_raw_SAM_rows_including_AUX_order",rawDifferent);
        report.add("core_field_different_rows",new Gson().toJsonTree(coreDiff));report.add("typed_AUX_different_rows",new Gson().toJsonTree(auxDiff));
        report.add("typed_AUX_difference_kinds",new Gson().toJsonTree(auxKinds));
        report.addProperty("complete_fields_and_typed_AUX_equal",different==0&&leftCount>0&&leftCount==rightCount);return report;
    }
    private static JsonObject materialize(String[] args)throws Exception {
        Path output=Paths.get(args[3]);if(Files.exists(output))throw new IllegalStateException("Output already exists");
        SamReaderFactory factory=SamReaderFactory.makeDefault().validationStringency(ValidationStringency.SILENT).referenceSequence(new File(args[1]));
        long records=0;
        try(SamReader input=factory.open(new File(args[2]));
            SAMFileWriter writer=new SAMFileWriterFactory().setCreateIndex(false).makeBAMWriter(input.getFileHeader(),true,output.toFile())) {
            for(SAMRecord record:input){writer.addAlignment(record);records++;}
        }
        JsonObject report=new JsonObject();report.addProperty("status","materialized_original_GATK_HTSJDK_decoded_CRAM_once");
        report.addProperty("records",records);report.addProperty("alignment_or_quality_algorithm_changed",false);return report;
    }
    private static JsonObject tables(String[] args)throws Exception {
        String[] names={"Arguments","Quantized","RecalTable0","RecalTable1","RecalTable2"};
        String[][] columns={{"Argument","Value"},{"QualityScore","Count","QuantizedScore"},{"ReadGroup","EventType","EmpiricalQuality","EstimatedQReported","Observations","Errors"},{"ReadGroup","QualityScore","EventType","EmpiricalQuality","Observations","Errors"},{"ReadGroup","QualityScore","CovariateValue","CovariateName","EventType","EmpiricalQuality","Observations","Errors"}};
        GATKReport a=new GATKReport(new File(args[1])),b=new GATKReport(new File(args[2]));JsonObject report=new JsonObject(),tables=new JsonObject();boolean equal=true;
        for(int n=0;n<names.length;n++) {
            GATKReportTable x=a.getTable(names[n]),y=b.getTable(names[n]);List<String> xr=new ArrayList<>(),yr=new ArrayList<>();
            for(GATKReportTable table:Arrays.asList(x,y)) {
                if(table.getNumColumns()!=columns[n].length)throw new IllegalStateException("Unexpected report column count");
                List<String> rows=table==x?xr:yr;
                for(int row=0;row<table.getNumRows();row++){List<String> values=new ArrayList<>();for(String col:columns[n])values.add(String.valueOf(table.get(row,col)));rows.add(new Gson().toJson(values));}
                Collections.sort(rows);
            }
            JsonObject t=new JsonObject();t.addProperty("rows_left",xr.size());t.addProperty("rows_right",yr.size());t.addProperty("all_declared_cells_equal",xr.equals(yr));t.add("columns",new Gson().toJsonTree(columns[n]));tables.add(names[n],t);equal&=xr.equals(yr);
        }
        report.addProperty("status","parsed_complete_BQSR_report");report.add("tables",tables);report.addProperty("all_five_tables_equal",equal);return report;
    }
    public static void main(String[] args)throws Exception {
        JsonObject report;Path output;
        if(args.length==7&&args[0].equals("align")){report=align(args);output=Paths.get(args[6]);}
        else if(args.length==4&&args[0].equals("report")){report=tables(args);output=Paths.get(args[3]);}
        else if(args.length==5&&args[0].equals("materialize")){report=materialize(args);output=Paths.get(args[4]);}
        else throw new IllegalArgumentException("align REF DICT LEFT RIGHT ROLES_JSON OUT_JSON | report LEFT RIGHT OUT_JSON | materialize REF CRAM BAM OUT_JSON");
        report.addProperty("raw_input_values_exported",false);
        String text=new GsonBuilder().setPrettyPrinting().create().toJson(report)+"\n";
        Files.write(output,text.getBytes(StandardCharsets.UTF_8),StandardOpenOption.CREATE_NEW);System.out.println(text);
    }
}
