"""Build and invoke a minimal same-Conda GATK QC CLI lifecycle adapter.

Only the standard library is imported. The original jar, scientific options,
tool lifecycle, and exception exit codes are retained. No scientific algorithm
is replaced, and no external dependency or environment is modified.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import shutil

from ._preprocessing_io import atomic, bound, digest, identity


QC_TOOLS = frozenset(("CollectMultipleMetrics", "CollectAlignmentSummaryMetrics", "CollectInsertSizeMetrics"))
QC_MAIN = "org.fudan.wgs.preprocessing.QcMain"
DEFAULT_JAVA_OPTIONS = (
    "-Xmx4g",
    "-Dsamjdk.use_async_io_read_samtools=false",
    "-Dsamjdk.use_async_io_write_samtools=true",
    "-Dsamjdk.use_async_io_write_tribble=false",
    "-Dsamjdk.compression_level=2",
)


def build_preprocessing_runtime(conda_prefix, build_dir, *, cpu_affinity=None,
                                host_memory_limit_gib=4, host_memory_reserve_gib=4):
    """Compile a new QC adapter with the selected read-only Conda JDK 8.

    ``build_dir`` must be new and outside the Conda prefix. The default is one
    permitted CPU, a 4 GiB sampled task-PSS budget, and 4 GiB live reserve. The
    artifact is a runtime candidate; compiling it does not validate real data.
    """
    from .single_preprocessing import _limits, _Runner

    prefix = Path(conda_prefix).resolve(strict=True)
    output = Path(build_dir).resolve()
    source = Path(__file__).with_name("_preprocessing_runtime_assets") / "QcMain.java"
    if not (prefix / "conda-meta").is_dir():
        raise ValueError("conda_prefix must be an existing Conda environment")
    if output.exists() or output == prefix or prefix in output.parents or output in prefix.parents:
        raise ValueError("build_dir must be new and outside the Conda prefix")
    if source.parent in output.parents or output == source.parent:
        raise ValueError("build_dir must be outside the packaged source directory")
    for value in (host_memory_limit_gib, host_memory_reserve_gib):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Build memory budget and reserve must be finite positive GiB")
    limits = _limits()
    allowed = os.sched_getaffinity(0)
    cpus = sorted(allowed)[:1] if cpu_affinity is None else list(cpu_affinity)
    if (not cpus or len(cpus) > 4 or len(set(cpus)) != len(cpus)
            or any(type(cpu) is not int for cpu in cpus) or not set(cpus) <= allowed):
        raise ValueError("cpu_affinity must contain one to four distinct allowed CPUs")
    if (len(cpus) > limits["cpu_quota_count"]
            or any(len(cpus) > quota for quota in limits["finite_cpu_quotas"])):
        raise ValueError("Requested build CPUs exceed the actual ancestor CPU quota")
    required = int((host_memory_limit_gib + host_memory_reserve_gib) * 2**30)
    if min(limits["available_bytes"], limits["admission_available_bytes"]) < required:
        raise MemoryError("QC runtime build lacks the requested actual/admission headroom")
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 64 * 2**20:
        raise OSError("QC runtime build requires 64 MiB of free storage")
    tools = {name: prefix / "bin" / name for name in ("java", "javac", "jar")}
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in tools.values()):
        raise ValueError("The selected Conda environment must contain Java/Javac/Jar")
    jars = list((prefix / "share").glob("*/gatk-package-4.2.0.0-local.jar"))
    if len(jars) != 1:
        raise ValueError("Exactly one GATK 4.2.0.0 local jar is required in the Conda prefix")
    gatk = jars[0].resolve(strict=True)
    if prefix not in gatk.parents:
        raise ValueError("The GATK jar must remain inside the selected Conda prefix")
    tool_bindings = {name: bound(path) for name, path in tools.items()}
    jar_binding = bound(gatk)
    source_binding = bound(source)
    package_records = {p.name: digest(p) for p in (prefix / "conda-meta").glob("*.json")}
    previous_umask = os.umask(0o077)
    try:
        output.mkdir(parents=True)
    finally:
        os.umask(previous_umask)
    snapshot = output / "source" / "QcMain.java"
    snapshot.parent.mkdir()
    shutil.copyfile(source, snapshot)
    if digest(snapshot) != source_binding["sha256"]:
        raise RuntimeError("Copied QC runtime source SHA-256 differs")
    config = dict(schema="fudan-QC-runtime-build-v1", cpu_affinity=cpus,
                  host_memory_limit_gib=host_memory_limit_gib,
                  host_memory_reserve_gib=host_memory_reserve_gib,
                  source=source_binding, builder_sha256=digest(__file__),
                  tools=tool_bindings, gatk_jar=jar_binding,
                  Conda_package_records_sha256=package_records)
    runner = _Runner(output, prefix, config, False)
    runner.env["CUDA_VISIBLE_DEVICES"] = ""
    classes = output / "classes"; classes.mkdir()
    runtime = output / "fudan-qc-runtime.jar"
    try:
        runner.stage("javac", [[str(tools["javac"]), "-proc:none", "-source", "8", "-target", "8",
                               "-cp", str(gatk), "-d", str(classes), str(snapshot)]],
                     [classes / "org/fudan/wgs/preprocessing/QcMain.class"], host_memory_limit_gib)
        runner.stage("jar", [[str(tools["jar"]), "cf", str(runtime), "-C", str(classes), "org"]],
                     [runtime], host_memory_limit_gib)
        for name, original in tool_bindings.items():
            if bound(tools[name]) != original:
                raise RuntimeError("Conda Java tool changed during the build")
        if bound(gatk) != jar_binding or bound(source) != source_binding:
            raise RuntimeError("The GATK jar or project source changed during the build")
        if package_records != {p.name: digest(p) for p in (prefix / "conda-meta").glob("*.json")}:
            raise RuntimeError("Conda package records changed during the build")
        artifacts = dict(runtimeJar=runtime, gatkJar=gatk, java=tools["java"])
        assets = dict(schema="fudan-QC-runtime-assets-v1", status="compiled_not_scientific_validation",
                      condaPrefix=str(prefix), mainClass=QC_MAIN,
                      **{key: str(path) for key, path in artifacts.items()},
                      sha256={key: digest(path) for key, path in artifacts.items()},
                      identities={key: identity(path) for key, path in artifacts.items()},
                      source_sha256=source_binding["sha256"], builder_sha256=digest(__file__),
                      real_data_analysis_started=False)
        receipt = dict(status=assets["status"], stages=runner.events, artifacts=assets,
                       Conda_package_records_unchanged=True, scientific_validation_assessed=False)
        atomic(output / "build_receipt.private.json", receipt)
        assets["build_receipt"] = str(output / "build_receipt.private.json")
        assets["build_receipt_sha256"] = digest(output / "build_receipt.private.json")
        atomic(output / "assets.private.json", assets)
        return assets
    except BaseException as error:
        atomic(output / "build_failed.private.json", dict(status="failed", error_type=type(error).__name__,
                                                         error=str(error), stages=runner.events))
        raise


def qc_argv(assets, tool_name, *tool_args, java_options=()):
    """Build a shell-free argv preserving GATK's original samjdk defaults.

    ``java_options`` is the already shlex-split original ``--java-options``
    sequence. The CLI exits normally after the original tool has returned.
    """
    if tool_name not in QC_TOOLS:
        raise ValueError("The QC runtime adapter is restricted to the three released QC commands")
    if assets.get("schema") != "fudan-QC-runtime-assets-v1" or assets.get("mainClass") != QC_MAIN:
        raise ValueError("Invalid QC runtime assets")
    if not isinstance(java_options, (list, tuple)) or any(not isinstance(x, str) or not x.startswith("-")
            or x in ("-jar", "-cp", "-classpath", "--class-path") for x in java_options):
        raise ValueError("java_options must contain explicit JVM options, not classpath or launcher changes")
    receipt_path = Path(assets["build_receipt"])
    if digest(receipt_path) != assets["build_receipt_sha256"]:
        raise RuntimeError("QC runtime build receipt SHA-256 changed")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "compiled_not_scientific_validation":
        raise RuntimeError("QC runtime build did not commit successfully")
    core={key:value for key,value in assets.items() if key not in ("build_receipt","build_receipt_sha256")}
    if receipt.get("artifacts") != core:
        raise RuntimeError("QC runtime assets differ from the committed build")
    for name, expected in assets["identities"].items():
        if identity(assets[name]) != expected:
            raise RuntimeError("QC runtime artifact identity changed: " + name)
    return [assets["java"], *DEFAULT_JAVA_OPTIONS, *java_options, "-cp",
            assets["runtimeJar"] + ":" + assets["gatkJar"], QC_MAIN,
            tool_name, *map(str, tool_args)]
