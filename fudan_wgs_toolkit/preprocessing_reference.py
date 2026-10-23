"""One immutable, shared reference check for original-software preprocessing.

The original samtools dictionary is checked during cohort preparation. Reuse requires unchanged
file identities (including ctime), the exact tool and helper, and a committed
receipt. No reference, index or Conda environment is rewritten.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from ._preprocessing_io import atomic, bound, digest, identity


def dictionary_sequences(path):
    rows, names = [], set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith('@SQ\t'):
                continue
            pieces = [field.split(':', 1) for field in line.rstrip('\r\n').split('\t')[1:]]
            if any(len(field) != 2 for field in pieces):
                raise ValueError('Malformed reference dictionary')
            fields = dict(pieces)
            if len(fields) != len(pieces):
                raise ValueError('Duplicate dictionary field')
            name, length, md5 = fields.get('SN'), fields.get('LN'), fields.get('M5')
            if not name or name in names or not length or not length.isdecimal() or int(length) < 1 or not re.fullmatch('[0-9a-fA-F]{32}', md5 or ''):
                raise ValueError('Dictionary requires unique names, positive lengths and sequence MD5')
            names.add(name)
            rows.append((name, length, md5.lower()))
    if not rows:
        raise ValueError('Empty reference dictionary')
    return rows


def _check_axes(reference, computed):
    expected = dictionary_sequences(reference.with_suffix('.dict'))
    if dictionary_sequences(computed) != expected:
        raise ValueError('FASTA sequence content differs from the supplied dictionary')
    with Path(str(reference)+'.fai').open() as stream:
        fai = [tuple(line.rstrip('\r\n').split('\t')[:2]) for line in stream]
    if fai != [row[:2] for row in expected]:
        raise ValueError('FAI differs from the reference sequence axis')
    return len(expected)


def _bound_and_check_fai(reference):
    """Hash and check every wrapped FASTA line against all five FAI columns.

    This optional cache accepts wrapped sequence lines up to 16 MiB. An
    oversized/unusual FASTA is rejected explicitly; the ordinary original-tool
    entry remains available. No chromosome is materialized in memory.
    """
    before = identity(reference)
    sha, rows, seen = hashlib.sha256(), [], set()
    name = None
    length = offset = bases = width = 0
    terminal_line = False
    with reference.open('rb') as stream:
        while True:
            position = stream.tell()
            line = stream.readline(16*2**20+1)
            if not line:
                break
            if len(line) > 16*2**20:
                raise ValueError('Reference validation cache requires wrapped FASTA lines <=16 MiB')
            sha.update(line)
            if line.startswith(b'>'):
                if name is not None:
                    if not length:raise ValueError('Empty FASTA contig')
                    rows.append((name,str(length),str(offset),str(bases),str(width)))
                tokens = line[1:].split()
                if not tokens:raise ValueError('Missing FASTA contig name')
                name = tokens[0].decode('utf-8')
                if name in seen:raise ValueError('Duplicate FASTA contig')
                seen.add(name)
                length = bases = width = 0
                terminal_line = False
                continue
            if name is None or terminal_line:
                raise ValueError('FASTA sequence line cannot be represented by its FAI')
            sequence = line.rstrip(b'\r\n')
            if not sequence or re.fullmatch(b'[A-Za-z*.-]+',sequence) is None:
                raise ValueError('Unusual FASTA sequence line; reference cache not admitted')
            if not bases:
                offset, bases, width = position, len(sequence), len(line)
            elif len(sequence) > bases or (len(sequence)==bases and len(line)!=width and line.endswith(b'\n')):
                raise ValueError('Inconsistent FASTA wrapping')
            terminal_line = len(sequence)<bases or not line.endswith(b'\n')
            length += len(sequence)
    if name is None or not length:raise ValueError('Empty FASTA reference')
    rows.append((name,str(length),str(offset),str(bases),str(width)))
    with Path(str(reference)+'.fai').open() as stream:
        actual = [tuple(line.rstrip('\r\n').split('\t')[:5]) for line in stream]
    if actual != rows:raise ValueError('FAI offsets or line geometry differ from the actual FASTA')
    if identity(reference)!=before:raise RuntimeError('Reference changed during full validation')
    return dict(before,sha256=sha.hexdigest())


def verify_preprocessing_reference(assets, *, reference_fasta, conda_prefix):
    """Verify a committed reference check without rescanning the FASTA.

    This is an identity-based reuse policy, not a new full-content hash. The
    one-time receipt retains the original full SHA and per-sequence MD5 check.
    Changes to inode, device, size, mtime or ctime invalidate reuse.
    """
    if isinstance(assets, (str, os.PathLike)):
        assets = json.loads(Path(assets).read_text())
    if not isinstance(assets, dict) or assets.get('schema') != 1 or assets.get('status') != 'reference_verified':
        raise ValueError('Expected a committed reference validation asset record')
    reference = Path(reference_fasta).resolve(strict=True)
    samtools = (Path(conda_prefix).resolve(strict=True)/'bin/samtools').resolve(strict=True)
    record = assets.get('receipt')
    if not isinstance(record, dict) or bound(record['path']) != record:
        raise ValueError('Reference validation receipt changed')
    receipt = json.loads(Path(record['path']).read_text())
    expected = {key: value for key, value in assets.items() if key != 'receipt'}
    if receipt.get('status') != 'reference_verified' or receipt.get('assets') != expected:
        raise ValueError('Reference validation asset and receipt disagree')
    if assets.get('helper_sha256') != digest(__file__):
        raise ValueError('Reference validation helper changed')
    if assets.get('fai_geometry_verified') is not True:
        raise ValueError('Reference cache lacks a complete FAI geometry check')
    inputs = assets.get('inputs', {})
    paths = [reference, Path(str(reference)+'.fai'), reference.with_suffix('.dict')]
    if set(inputs) != {str(path) for path in paths}:
        raise ValueError('Reference validation is bound to a different reference')
    for path in paths:
        saved = inputs[str(path)]
        if identity(path) != {key: value for key, value in saved.items() if key != 'sha256'}:
            raise ValueError('Reference input identity changed; prepare a new validation')
        # Small sidecars are fully hashed on every reuse.
        if path != reference and bound(path) != saved:
            raise ValueError('Reference sidecar content changed')
    if assets.get('samtools') != bound(samtools):
        raise ValueError('Reference validation tool changed')
    dictionary = assets.get('computed_dictionary')
    if not isinstance(dictionary, dict) or bound(dictionary['path']) != dictionary:
        raise ValueError('Computed reference dictionary changed')
    _check_axes(reference, Path(dictionary['path']))
    return assets


def prepare_preprocessing_reference(reference_fasta, conda_prefix, output_dir, *,
                                    cpu_affinity=None, host_memory_limit_gib=2,
                                    host_memory_reserve_gib=4, resume=False):
    """Prepare one reference check for several single_subj_preprocessing runs.

    Inputs: plain FASTA, adjacent .fai and .dict, and an installed samtools in
    conda_prefix/bin. Output: private assets/receipt JSON and original samtools
    recomputed dictionary. The FASTA/sidecars/tools remain unchanged. Include
    this one-time preparation in cohort end-to-end timing. Preparation makes
    two full passes: SHA/FAI geometry and original samtools sequence MD5.
    """
    from .single_preprocessing import _Runner, _limits
    started = time.monotonic()
    if type(resume) is not bool:
        raise ValueError('resume must be boolean')
    for value in (host_memory_limit_gib, host_memory_reserve_gib):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Reference budgets must be finite positive GiB')
    reference = Path(reference_fasta).resolve(strict=True)
    if reference.name.endswith(('.gz', '.bgz')):
        raise ValueError('Reference validation requires a plain FASTA')
    prefix = Path(conda_prefix).resolve(strict=True)
    if not (prefix/'conda-meta').is_dir():
        raise ValueError('Reference validation requires an existing Conda environment')
    tool = (prefix/'bin/samtools').resolve(strict=True)
    allowed = set(os.sched_getaffinity(0))
    cpus = [min(allowed)] if cpu_affinity is None else list(cpu_affinity)
    if not cpus or len(cpus) != len(set(cpus)) or any(type(cpu) is not int or cpu not in allowed for cpu in cpus):
        raise ValueError('Reference CPU affinity must be distinct allowed CPUs')
    limits = _limits()
    if len(cpus) > limits['cpu_quota_count'] or any(len(cpus)>quota for quota in limits['finite_cpu_quotas']):
        raise ValueError('Reference CPU affinity exceeds the actual CPU quota')
    root = Path(output_dir).resolve()
    if root == prefix or prefix in root.parents:
        raise ValueError('Reference output must be outside the Conda environment')
    if any(root==path.resolve() or root in path.resolve().parents for path in
           (reference,Path(str(reference)+'.fai'),reference.with_suffix('.dict'),tool)):
        raise ValueError('Reference validation output must not contain any input or tool')
    if host_memory_limit_gib < 1:
        raise ValueError('Reference memory budget must be at least 1 GiB')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError('Reference validation output must be private and owned')
    with (root/'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asset_file = root/'assets.private.json'
        if asset_file.exists():
            if not resume:
                raise ValueError('Existing reference validation requires resume=True')
            return verify_preprocessing_reference(asset_file, reference_fasta=reference, conda_prefix=prefix)
        paths = [reference, Path(str(reference)+'.fai'), reference.with_suffix('.dict')]
        # Bind sidecars before the geometry scan consumes the FAI. The final
        # binding check then brackets all geometry and original-tool reads.
        inputs = {str(path): bound(path) for path in paths if path!=reference}
        inputs[str(reference)] = _bound_and_check_fai(reference)
        tool_binding = bound(tool)
        config = dict(schema=1, inputs=inputs, samtools=tool_binding,
                      helper_sha256=digest(__file__), cpu_affinity=sorted(cpus),
                      host_memory_limit_gib=host_memory_limit_gib,
                      host_memory_reserve_gib=host_memory_reserve_gib)
        runner = _Runner(root, prefix, config, resume)
        computed = root/'reference.recomputed.dict'
        runner.stage('reference_dictionary', [[str(tool), 'dict', '-o', str(computed), str(reference)]], [computed], 1)
        contigs = _check_axes(reference, computed)
        if (any(identity(path) != {key:value for key,value in inputs[str(path)].items() if key!='sha256'} for path in paths)
                or any(bound(path)!=inputs[str(path)] for path in paths if path!=reference)
                or bound(tool) != tool_binding):
            raise RuntimeError('Reference or tool changed during preparation')
        assets = dict(schema=1, status='reference_verified', inputs=inputs,
                      samtools=tool_binding, helper_sha256=digest(__file__),
                      computed_dictionary=bound(computed), contigs=contigs,
                      fai_geometry_verified=True,
                      reuse_policy='unchanged_physical_file_identity_with_ctime',
                      full_content_hash_repeated_on_reuse=False)
        receipt = root/'reference_validation.private.json'
        atomic(receipt, dict(status='reference_verified', assets=assets,
                             elapsed_seconds=time.monotonic()-started,
                             stages=runner.events))
        result = dict(assets, receipt=bound(receipt))
        atomic(asset_file, result)
        return verify_preprocessing_reference(result, reference_fasta=reference, conda_prefix=prefix)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=prepare_preprocessing_reference.__doc__)
    parser.add_argument('--config',required=True,help='JSON object using the Python parameter names')
    args=parser.parse_args(argv)
    result=prepare_preprocessing_reference(**json.loads(Path(args.config).read_text()))
    print(json.dumps({'status':result['status'],'contigs':result['contigs'],
                      'reuse_policy':result['reuse_policy']}))


if __name__=='__main__':
    main()
