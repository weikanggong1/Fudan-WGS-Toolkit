"""Build the optional, repaired PairHMM adapter with one existing Conda prefix.

This module uses the Python standard library only. It never installs software,
modifies the prefix, downloads inputs, or starts a real-data analysis. The CPU
HaplotypeCaller command remains the default; optional CUDA execution is explicit.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


PROVIDER = "head-native-cuda-pairhmm-fp32-wavefront-packed-H2D-initfix-v1b"
GKL_FIXTURE_SHA256 = "598c5a3cedb9aaa10a0da258a9d6272ef51e4b59262dea90293fe0c6d9a86244"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path):
    stat = Path(path).stat()
    return {key: getattr(stat, "st_" + key) for key in
            ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_preprocessing_gpu(conda_prefix, build_dir, *, gatk_jar=None,
                            cpu_affinity=None, gpu_selftest_uuid=None,
                            public_fixture=None, public_fixture_sha256=None):
    """Return JSON-serializable classpath/nativeLib/hash assets after compilation.

    ``build_dir`` must be new and outside ``conda_prefix``. ``cpu_affinity`` is
    one to four allowed Linux CPU IDs. CUDA is compiled for sm_80 (A100).
    The build uses a fixed 4 GiB sampled process-PSS budget and requires another
    4 GiB of live memory headroom, plus 4 GiB of free build storage. Requested
    CPUs must fit the actual finite ancestor quota. These are admission and
    monitoring checks, rather than a cgroup memory reservation or hard cap.
    Supplying ``gpu_selftest_uuid`` explicitly enables numerical GPU selftests;
    it is not permission to process FASTQ/BAM files. An optional official GKL
    0.8.6 fixture must already exist and match its fixed upstream SHA-256.
    """
    prefix = Path(conda_prefix).resolve(strict=True)
    output = Path(build_dir).resolve()
    bundle = Path(__file__).resolve().parent / "_preprocessing_gpu"
    if not (prefix / "conda-meta").is_dir():
        raise ValueError("conda_prefix must be an existing Conda environment")
    if output.exists() or output == prefix or prefix in output.parents or output in prefix.parents:
        raise ValueError("build_dir must be new and outside the Conda prefix")
    if output == bundle or bundle in output.parents:
        raise ValueError("build_dir must be outside the packaged source directory")
    # Import at call time: this is also used before the analysis Runner exists.
    # The resource helpers use the standard library and never import Torch.
    if __package__:
        from .single_preprocessing import _limits, _pss, _descendants, _cleanup_stage, _group_members
    else:
        # Preserve the documented direct-script CLI as well as ``python -m``.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from fudan_wgs_toolkit.single_preprocessing import _limits, _pss, _descendants, _cleanup_stage, _group_members
    build_budget = 4 * 2**30
    memory_reserve = 4 * 2**30
    limits = _limits()
    cpus = (list(cpu_affinity) if cpu_affinity is not None else
            sorted(os.sched_getaffinity(0))[:min(4, limits['cpu_quota_count'])])
    if (not cpus or len(cpus) > 4 or len(set(cpus)) != len(cpus)
            or any(type(cpu) is not int for cpu in cpus)
            or not set(cpus) <= os.sched_getaffinity(0)):
        raise ValueError("cpu_affinity must contain one to four distinct allowed CPU IDs")
    if (len(cpus) > limits['cpu_quota_count']
            or any(len(cpus) > quota for quota in limits['finite_cpu_quotas'])):
        raise ValueError("Requested build CPUs exceed the actual ancestor CPU quota")

    def admission():
        current = _limits()
        if (current['admission_available_bytes'] < build_budget + memory_reserve
                or current['available_bytes'] < build_budget + memory_reserve):
            raise MemoryError("GPU build requires 4 GiB prospective PSS plus 4 GiB live headroom")
        if (len(cpus) > current['cpu_quota_count']
                or any(len(cpus) > quota for quota in current['finite_cpu_quotas'])):
            raise RuntimeError("The finite CPU quota no longer admits the requested build CPUs")
        return current

    limits = admission()
    disk_parent = output.parent
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    free_build_bytes = shutil.disk_usage(disk_parent).free
    if free_build_bytes < build_budget:
        raise OSError("GPU build requires 4 GiB free storage in its output filesystem")
    if gpu_selftest_uuid is not None:
        if (not isinstance(gpu_selftest_uuid, str) or not gpu_selftest_uuid.startswith("GPU-")
                or "," in gpu_selftest_uuid):
            raise ValueError("Select exactly one explicitly authorized physical GPU UUID")
    if public_fixture is not None and gpu_selftest_uuid is None:
        raise ValueError("The official fixture test requires an explicit GPU selftest UUID")
    if public_fixture_sha256 is not None and public_fixture_sha256 != GKL_FIXTURE_SHA256:
        raise ValueError("The supplied official GKL 0.8.6 fixture hash differs")
    fixture = Path(public_fixture).resolve(strict=True) if public_fixture is not None else None
    if fixture is not None and _sha(fixture) != GKL_FIXTURE_SHA256:
        raise ValueError("The official GKL 0.8.6 fixture SHA-256 differs")
    manifest_path = bundle / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("provider") != PROVIDER or not manifest.get("files"):
        raise RuntimeError("Missing repaired packed source manifest")
    tools = {name: prefix / "bin" / executable for name, executable in (
        ("CC", "x86_64-conda-linux-gnu-gcc"), ("CXX", "x86_64-conda-linux-gnu-g++"),
        ("CUDACXX", "nvcc"), ("cmake", "cmake"), ("ninja", "ninja"),
        ("java", "java"), ("javac", "javac"), ("jar", "jar"))}
    if any(not path.is_file() or not os.access(path, os.X_OK) for path in tools.values()):
        raise RuntimeError("Compiler, CUDA, CMake/Ninja and Java tools must exist in this prefix")
    jars = [Path(gatk_jar).resolve(strict=True)] if gatk_jar else list(
        (prefix / "share").glob("*/gatk-package-4.2.0.0-local.jar"))
    if len(jars) != 1 or prefix not in jars[0].parents:
        raise ValueError("Select the GATK 4.2.0.0 local jar inside this Conda prefix")
    jar = jars[0]
    previous_umask = os.umask(0o077)
    try:
        output.mkdir(parents=True)
    finally:
        os.umask(previous_umask)
    snapshot = output / "source"
    for relative, expected in manifest["files"].items():
        item = Path(relative)
        source = bundle / item
        if item.is_absolute() or ".." in item.parts or not item.parts or item.parts[0] != "assets":
            raise RuntimeError("Invalid source manifest entry")
        if source.is_symlink() or _sha(source) != expected:
            raise RuntimeError("Packaged source SHA-256 differs: " + relative)
        target = snapshot / item.relative_to("assets")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if _sha(target) != expected:
            raise RuntimeError("Copied source SHA-256 differs: " + relative)
    package_records = {p.name: _sha(p) for p in sorted((prefix / "conda-meta").glob("*.json"))}
    temporary = output / "tmp"
    temporary.mkdir()
    env = dict(os.environ, CONDA_PREFIX=str(prefix), JAVA_HOME=str(prefix),
               CC=str(tools["CC"]), CXX=str(tools["CXX"]), CUDAHOSTCXX=str(tools["CXX"]),
               CUDACXX=str(tools["CUDACXX"]), CUDAToolkit_ROOT=str(prefix),
               NVCC_PREPEND_FLAGS="--cudart=shared --cudadevrt=none",
               PATH=str(prefix / "bin") + ":/usr/bin:/bin", LD_LIBRARY_PATH=str(prefix / "lib"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES=gpu_selftest_uuid or "", TMPDIR=str(temporary),
               CUDA_CACHE_PATH=str(output / "cuda_cache"), XDG_CACHE_HOME=str(output / "cache"))
    receipt = dict(schema="fudan-preprocessing-gpu-build-v1", status="building",
                   started_utc=_utc(), source_sha256=manifest["files"],
                   source_manifest_sha256=_sha(manifest_path), builder_sha256=_sha(__file__), prefix=str(prefix),
                   package_record_sha256=package_records, cpu_affinity=cpus,
                   gatk_jar_sha256=_sha(jar), expected_provider=PROVIDER,
                   GPU_selftest_requested=gpu_selftest_uuid is not None,
                   real_data_analysis_started=False, steps=[],
                   resource_admission=limits, build_pss_budget_bytes=build_budget,
                   memory_reserve_bytes=memory_reserve, admitted_free_storage_bytes=free_build_bytes,
                   peak_sampled_pss_bytes=None, pss_is_sampled_not_a_cgroup_hard_limit=True)

    def save():
        temporary = output / "build_receipt.private.json.writing"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output / "build_receipt.private.json")

    def step(name, argv):
        started = time.monotonic()
        record = dict(name=name, status="running", admission=admission(),
                      exit_code=None, peak_sampled_pss_bytes=None, session_drained=False)
        receipt['steps'].append(record)
        save()
        process = None

        def setup():
            os.sched_setaffinity(0, cpus)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        try:
            with (output / (name + ".stdout.log")).open("wb") as stdout, (output / (name + ".stderr.log")).open("wb") as stderr, \
                    (output / (name + ".resources.private.jsonl")).open("w", buffering=1) as telemetry:
                process = subprocess.Popen(list(map(str, argv)), stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, cwd=output, env=env,
                                           start_new_session=True, preexec_fn=setup)
                last_limits = None
                next_limits = 0.0
                while process.poll() is None:
                    # Parent Python is part of the build's memory budget.
                    # Session membership also catches an adopted compiler child.
                    measured = _pss(_descendants(process.pid) | set(_group_members(process.pid)) | {os.getpid()})
                    if measured is None:
                        raise RuntimeError("Live build PSS is unavailable")
                    record['peak_sampled_pss_bytes'] = max(record['peak_sampled_pss_bytes'] or 0, measured)
                    receipt['peak_sampled_pss_bytes'] = max(receipt['peak_sampled_pss_bytes'] or 0, measured)
                    if measured > build_budget:
                        raise MemoryError("GPU build process PSS exceeded its fixed 4 GiB budget")
                    now = time.monotonic()
                    if now >= next_limits:
                        last_limits = _limits()
                        if last_limits['available_bytes'] < memory_reserve:
                            raise MemoryError("GPU build lost its 4 GiB live memory reserve")
                        if (len(cpus) > last_limits['cpu_quota_count']
                                or any(len(cpus) > quota for quota in last_limits['finite_cpu_quotas'])):
                            raise RuntimeError("CPU quota changed during GPU build")
                        next_limits = now + 1.0
                    telemetry.write(json.dumps(dict(elapsed_seconds=now-started, pss_bytes=measured,
                                                    live_headroom_bytes=last_limits['available_bytes'],
                                                    observed_cpu_quota_count=last_limits['cpu_quota_count'])) + '\n')
                    time.sleep(.25)
                record['exit_code'] = process.wait()
                if record['exit_code']:
                    raise RuntimeError("Build/component step failed: " + name)
                if _group_members(process.pid):
                    raise RuntimeError("Build step root exited before its session children drained")
                record.update(status='complete', session_drained=True)
                process = None
        except BaseException as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            if process is not None:
                try:
                    _cleanup_stage(process)
                    record['session_drained'] = not _group_members(process.pid)
                except BaseException as cleanup_error:
                    record['cleanup_error'] = str(cleanup_error)
                record['exit_code'] = process.poll()
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - started
            save()

    save()
    try:
        for name in ("CUDACXX", "CXX", "javac"):
            step(name + "_version", [tools[name], "-version" if name == "javac" else "--version"])
        native = output / "native_build"
        step("configure", [tools["cmake"], "-S", snapshot / "integration", "-B", native,
                           "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=80",
                           "-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared", "-DCMAKE_CUDA_FLAGS=--cudadevrt=none",
                           "-DCMAKE_CXX_COMPILER=" + str(tools["CXX"]),
                           "-DCMAKE_CUDA_COMPILER=" + str(tools["CUDACXX"]),
                           "-DCMAKE_CUDA_HOST_COMPILER=" + str(tools["CXX"]),
                           "-DCUDAToolkit_ROOT=" + str(prefix)])
        step("compile", [tools["cmake"], "--build", native, "--parallel", str(min(2, len(cpus)))])
        classes = output / "classes"; classes.mkdir()
        headers = output / "jni_headers"; headers.mkdir()
        sources = sorted((snapshot / "java").rglob("*.java"))
        step("javac", [tools["javac"], "-proc:none", "-source", "8", "-target", "8", "-cp", jar,
                       "-d", classes, "-h", headers, *sources])
        adapter = output / "fnit-pairhmm-adaptive.jar"
        probe = output / "fnit-pairhmm-probe.jar"
        step("jar_adapter", [tools["jar"], "cf", adapter, "-C", classes, "com"])
        step("jar_probe", [tools["jar"], "cf", probe, "-C", classes, "org"])
        step("adapter_classpath", [tools["java"], "-Dfnit.pairhmm.expectAdapter=true", "-cp",
                                  str(adapter) + ":" + str(probe) + ":" + str(jar),
                                  "org.fnit.wgs.PairHmmBatchProbe", "--class-only"])
        step("original_classpath", [tools["java"], "-cp", str(probe) + ":" + str(jar),
                                   "org.fnit.wgs.PairHmmBatchProbe", "--class-only"])
        selftest = native / "pairhmm_native/pairhmm_selftest"
        step("CPU_numerical_component", [selftest])
        if gpu_selftest_uuid is not None:
            step("GPU_numerical_component", [selftest, "--gpu"])
            data = json.loads((output / "GPU_numerical_component.stdout.log").read_text())
            if data["status"] != "passed" or not data["gpu_executed"] or data["dense_regression_pairs"] != 49152:
                raise RuntimeError("GPU numerical work was not verified")
        if fixture is not None:
            step("GPU_official_GKL086_fixture", [native / "pairhmm_native/pairhmm_public_fixture_test", fixture])
            data = json.loads((output / "GPU_official_GKL086_fixture.stdout.log").read_text())
            if data["status"] != "passed" or data["tested_pairs"] != 104 or data["failed_pairs"]:
                raise RuntimeError("The official 104-pair component did not pass")
            receipt["official_fixture_sha256"] = GKL_FIXTURE_SHA256
        artifacts = dict(adapterJar=adapter, probeJar=probe, nativeLib=native / "libfnit_pairhmm_jni.so",
                         cudaLibrary=native / "pairhmm_native/libpairhmm_cuda.so", gatkJar=jar)
        hashes = {key: _sha(path) for key, path in artifacts.items()}
        if package_records != {p.name: _sha(p) for p in sorted((prefix / "conda-meta").glob("*.json"))}:
            raise RuntimeError("Conda package records changed during the build")
        result = dict(schema="fudan-preprocessing-gpu-assets-v1", status="compiled_components_passed_not_WGS",
                      **{key: str(path) for key, path in artifacts.items()},
                      classpath=str(adapter) + ":" + str(jar), originalClasspath=str(jar),
                      replayClasspath=str(adapter) + ":" + str(probe) + ":" + str(jar),
                      originalReplayClasspath=str(probe) + ":" + str(jar),
                      condaPrefix=str(prefix), java=str(tools["java"]), provider=PROVIDER,
                      builder_sha256=_sha(__file__), source_manifest_sha256=_sha(manifest_path),
                      sha256=hashes, identities={key: _identity(path) for key, path in artifacts.items()},
                      default_backend="CPU_original_GKL", optional_GPU_backend="explicit_forceGPU_or_calibrated_auto",
                      GPU_numerical_selftest_passed=gpu_selftest_uuid is not None,
                      real_data_analysis_started=False, whole_WGS_pipeline_completed=False)
        receipt.update(status=result["status"], completed_utc=_utc(), artifacts=result,
                       Conda_package_records_unchanged=True)
        save()
        # An assets file is the final commit marker, only after the successful
        # build receipt is durable at its final name. Failed builds have none.
        assets_temporary = output / "assets.private.json.writing"
        assets_temporary.write_text(json.dumps(result, indent=2) + "\n")
        assets_temporary.replace(output / "assets.private.json")
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error), completed_utc=_utc())
        save()
        raise
    return result


def haplotypecaller_argv(assets, *, reference, bam, output, mode="cpu", audit=None,
                         native_threads=16, heap_gib=40, tmp_dir=None, intervals=(),
                         gpu_min_pairs=None, gpu_min_dp_cells=None, capture_max=None):
    """Create a shell-free HC argv. CPU is the default; no automatic GPU claim.

    The caller supplies identical BAM/reference/intervals for comparisons and
    applies its permitted CPU affinity. Use ``gpu_environment`` for execution.
    """
    if mode not in ("cpu", "forceGPU", "auto") or native_threads < 1 or heap_gib < 1:
        raise ValueError("Invalid backend, native thread count or heap size")
    for key, identity in assets["identities"].items():
        if _identity(assets[key]) != identity:
            raise RuntimeError("Built artifact identity changed: " + key)
    defaults = ["-Dsamjdk.use_async_io_read_samtools=false", "-Dsamjdk.use_async_io_write_samtools=true",
                "-Dsamjdk.use_async_io_write_tribble=false", "-Dsamjdk.compression_level=2"]
    argv = [assets["java"], *defaults, "-Xms" + str(heap_gib) + "G", "-Xmx" + str(heap_gib) + "G",
            "-XX:ParallelGCThreads=2"]
    if tmp_dir is not None:
        argv.append("-Djava.io.tmpdir=" + str(Path(tmp_dir).resolve()))
    classpath = assets["originalClasspath"]
    if mode != "cpu":
        if audit is None or not Path(audit).is_absolute():
            raise ValueError("Optional GPU execution requires an absolute private audit path")
        classpath = assets["classpath"]
        argv += ["-Dfnit.pairhmm.mode=" + mode, "-Dfnit.pairhmm.library=" + assets["nativeLib"],
                 "-Dfnit.pairhmm.device=0", "-Dfnit.pairhmm.audit=" + str(audit)]
        if mode == "auto":
            if (type(gpu_min_pairs) is not int or type(gpu_min_dp_cells) is not int
                    or gpu_min_pairs < 1 or gpu_min_dp_cells < 1):
                raise ValueError("auto requires explicit positive pair and DP-cell thresholds")
            argv += ["-Dfnit.pairhmm.gatkJar=" + assets["gatkJar"],
                     "-Dfnit.pairhmm.gatkJarSha256=" + assets["sha256"]["gatkJar"],
                     "-Dfnit.pairhmm.gpuMinPairs=" + str(gpu_min_pairs),
                     "-Dfnit.pairhmm.gpuMinDpCells=" + str(gpu_min_dp_cells)]
        if capture_max is not None:
            if not Path(capture_max).is_absolute():
                raise ValueError("Actual sequence capture requires an absolute private path")
            argv += ["-Dfnit.pairhmm.captureMax=" + str(capture_max)]
    argv += ["-cp", classpath, "org.broadinstitute.hellbender.Main", "HaplotypeCaller",
             "-R", str(reference), "-I", str(bam), "-O", str(output), "--emit-ref-confidence", "GVCF",
             "--native-pair-hmm-threads", str(native_threads), "--pair-hmm-implementation", "AVX_LOGLESS_CACHING_OMP"]
    for interval in intervals:
        argv += ["-L", str(interval)]
    return argv


def gpu_environment(assets, *, gpu_uuid=None, max_device_bytes=2 * 2**30, max_batch_pairs=128):
    """Return the same-prefix runtime environment; no selected GPU means hidden.

    The native allocation limit excludes the driver/context and other users.
    """
    if gpu_uuid is not None and (not gpu_uuid.startswith("GPU-") or "," in gpu_uuid):
        raise ValueError("Select one explicitly permitted GPU UUID")
    if not 0 < max_device_bytes <= 18 * 2**30 or not 0 < max_batch_pairs <= 65536:
        raise ValueError("Native allocation or batch limit is out of range")
    prefix = assets["condaPrefix"]
    return dict(os.environ, PATH=prefix + "/bin:/usr/bin:/bin", LD_LIBRARY_PATH=prefix + "/lib",
                JAVA_HOME=prefix, CONDA_PREFIX=prefix, CUDA_VISIBLE_DEVICES=gpu_uuid or "",
                FNIT_PAIRHMM_MAX_DEVICE_BYTES=str(max_device_bytes),
                FNIT_PAIRHMM_MAX_BATCH_PAIRS=str(max_batch_pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-prefix", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--gatk-jar")
    parser.add_argument("--cpu-affinity", help="One to four allowed CPU IDs separated by commas")
    parser.add_argument("--gpu-selftest-uuid")
    parser.add_argument("--public-fixture")
    parser.add_argument("--public-fixture-sha256")
    args = parser.parse_args()
    cpus = [int(value) for value in args.cpu_affinity.split(",")] if args.cpu_affinity else None
    result = build_preprocessing_gpu(args.conda_prefix, args.build_dir, gatk_jar=args.gatk_jar,
        cpu_affinity=cpus, gpu_selftest_uuid=args.gpu_selftest_uuid,
        public_fixture=args.public_fixture, public_fixture_sha256=args.public_fixture_sha256)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
