# Original GATK BQSR source patch

Upstream: [Broad Institute GATK](https://github.com/broadinstitute/gatk), version
4.2.0.0, commit `1f7fade2afbd706539eb983087907a0cb11ee734`.
The two source files are fetched from this fixed commit; their byte lengths and
SHA-256 values are recorded in `source_manifest.json` and verified before an
exact-context, two-file patch is applied. A previously downloaded source tree
can be supplied for an offline build; it must have the same original bytes.

- [BaseRecalibrationEngine.java](https://github.com/broadinstitute/gatk/blob/1f7fade2afbd706539eb983087907a0cb11ee734/src/main/java/org/broadinstitute/hellbender/utils/recalibration/BaseRecalibrationEngine.java)
- [ReadRecalibrationInfo.java](https://github.com/broadinstitute/gatk/blob/1f7fade2afbd706539eb983087907a0cb11ee734/src/main/java/org/broadinstitute/hellbender/utils/recalibration/ReadRecalibrationInfo.java)
- [Upstream LICENSE.TXT](https://github.com/broadinstitute/gatk/blob/1f7fade2afbd706539eb983087907a0cb11ee734/LICENSE.TXT)

The upstream source is Apache-2.0. Its exact license is included as
`LICENSE.GATK-4.2.0.0.txt` and in the generated overlay jar. The changed source
files carry a modification notice. Only the small production patch is packaged
here; complete upstream source, the installed original jar, original data and
private benchmark outputs are not redistributed. The project launcher and
builder follow this repository's GPL-3.0-only license. Dependencies in the
original GATK jar retain their own licenses. The pinned upstream root has no
NOTICE file; this is not a claim about notices inside dependency jars.

The patch caches the existing stateless read-transform composition in each
engine and omits two unused fractional indel-error arrays when the actual
constructed engine event axis contains only substitutions. The original
six-argument public constructor and the indel-enabled engine branch remain
available. Quality/error access to an omitted indel event explicitly fails.
Integer SNP/indel detection, BAQ, substitution-error math, table updates and
original global finalization are retained. Each parallel worker owns a separate
original engine. No cross-worker engine or model cache is introduced.

`SourcePatchedBqsrMain` verifies the CodeSource and exact class-resource SHA of
all emitted overlay classes in the same JVM that will train BQSR, before
forwarding to original `BaseRecalibrator` or the separately bound existing
parallel adapter. It also verifies original GATK Main. Its audit proves class
loading; scientific success additionally requires the committed training stage.
A probe/build success is never a real-data precision or performance result.

The engine's pre-existing object graph is not Java-serializable in the tested
original version (`CovariateKeyCache`); transient cache reconstruction is not
claimed as an assessed serialization feature. Scientific tests and private
run evidence are distinct from the public build/argument contracts.
