"""Streaming single-sample VCF/BED validation; no phenotype or annotation creation."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise ValueError('Expected a regular file')
    return dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def bound(path):
    before = identity(path)
    result = dict(before, sha256=digest(path))
    if identity(path) != before:
        raise RuntimeError('File changed during hashing')
    return result


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.writing.' + str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def vcf_oracle(vcf, sample_id, output_dir):
    """One bounded VCF scan, independent of PLINK, writing the expected BED axis.

    BED stores hard diploid genotypes, not phase or INFO/FORMAT quality fields.
    The complete VCF is retained. Only autosomes enter Toolkit's PLINK input.
    """
    target = Path(output_dir)
    target.mkdir()
    counts, excluded = {}, {}
    previous_chrom = None
    seen_chroms, seen_alleles = set(), set()
    previous_pos = -1
    maximum_allele_length=1
    samples = None
    with gzip.open(vcf, 'rt') as src, (target/'axis.tsv').open('w') as axis, (target/'expected.bed').open('wb') as bed:
        bed.write(b'\x6c\x1b\x01')
        for line in src:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM\t'):
                if samples is not None:
                    raise ValueError('Duplicate VCF sample header')
                samples = line.rstrip('\r\n').split('\t')[9:]
                if samples != [sample_id]:
                    raise ValueError('VCF must contain exactly the requested single sample')
                continue
            if line.startswith('#'):
                continue
            values = line.rstrip('\r\n').split('\t')
            if samples is None or len(values) != 10:
                raise ValueError('Malformed single-sample VCF')
            chrom = values[0][3:] if values[0].startswith('chr') else values[0]
            if chrom not in {str(i) for i in range(1, 23)}:
                excluded[values[0]] = excluded.get(values[0], 0) + 1
                continue
            pos, ref, alt = int(values[1]), values[3], values[4]
            if pos < 1 or not re.fullmatch('[ACGTNacgtn]+', ref) or not alt or alt=='.' or ',' in alt or re.search(r'\s',alt) or ref == alt:
                raise ValueError('BED requires normalized biallelic alleles; VCF is retained')
            maximum_allele_length=max(maximum_allele_length,len(ref),len(alt))
            if chrom != previous_chrom:
                if chrom in seen_chroms:
                    raise ValueError('Non-contiguous VCF chromosome order')
                seen_chroms.add(chrom)
                previous_chrom, previous_pos = chrom, -1
                seen_alleles.clear()
            if pos < previous_pos:
                raise ValueError('VCF positions must be ordered')
            if pos != previous_pos:
                seen_alleles.clear()
            if (ref, alt) in seen_alleles:
                raise ValueError('Duplicate variant identity')
            previous_pos = pos; seen_alleles.add((ref, alt))
            keys, fields = values[8].split(':'), values[9].split(':')
            if keys.count('GT') != 1 or keys.index('GT') >= len(fields):
                raise ValueError('Missing hard-call GT')
            gt = fields[keys.index('GT')]
            if gt in ('.', './.', '.|.'):
                code = 1
            elif re.fullmatch('[01][/|][01]', gt):
                code = (3, 2, 0)[int(gt[0]) + int(gt[2])]
            else:
                raise ValueError('Partial, haploid or non-biallelic GT cannot be silently coerced')
            bed.write(bytes([code]))  # One subject; high six bits are zero padding.
            axis.write('\t'.join((chrom, str(pos), ref, alt)) + '\n')
            counts[chrom] = counts.get(chrom, 0) + 1
    if samples is None:
        raise ValueError('Missing VCF sample header')
    return dict(chromosome_counts=counts, excluded_non_autosomal_records=excluded,
                variants=sum(counts.values()), samples=1,maximum_allele_length=maximum_allele_length,
                retained='hard diploid GT, missing state, CHR/POS/REF/ALT and order',
                not_encoded_in_bed=['phase', 'INFO', 'QUAL', 'FILTER', 'FORMAT fields other than GT'])


def verify_and_split_plink(prefix, oracle_dir, output_dir, sample_id, counts):
    """Full readback plus bounded chromosome partition; no 22 repeated VCF scans."""
    prefix, oracle_dir, output_dir = Path(prefix), Path(oracle_dir), Path(output_dir)
    fam = Path(str(prefix)+'.fam').read_text().splitlines()
    if len(fam) != 1 or fam[0].split()[:2] != [sample_id, sample_id]:
        raise ValueError('PLINK FID/IID differ from the true sample ID')
    output_dir.mkdir()
    handles = {}
    rows = 0
    try:
        with Path(str(prefix)+'.bed').open('rb') as bed, (oracle_dir/'expected.bed').open('rb') as expected, Path(str(prefix)+'.bim').open() as bim, (oracle_dir/'axis.tsv').open() as axis:
            if bed.read(3) != b'\x6c\x1b\x01' or expected.read(3) != b'\x6c\x1b\x01':
                raise ValueError('Invalid variant-major BED header')
            for variant in axis:
                chrom, pos, ref, alt = variant.rstrip('\n').split('\t')
                row = bim.readline().split()
                identifier = ':'.join((chrom, pos, ref, alt))
                if len(row) != 6 or [row[0], row[1], row[3], row[4], row[5]] != [chrom, identifier, pos, alt, ref]:
                    raise ValueError('PLINK allele direction, order or variant identity changed')
                byte = bed.read(1)
                if byte != expected.read(1) or len(byte) != 1:
                    raise ValueError('PLINK hard call or BED padding changed')
                if chrom not in handles:
                    chunk = output_dir/('chr'+chrom)
                    bh, mh = Path(str(chunk)+'.bed').open('wb'), Path(str(chunk)+'.bim').open('w')
                    bh.write(b'\x6c\x1b\x01')
                    Path(str(chunk)+'.fam').write_text(fam[0]+'\n')
                    handles[chrom] = bh, mh
                handles[chrom][0].write(byte)
                handles[chrom][1].write('\t'.join(row)+'\n')
                rows += 1
            if bed.read(1) or expected.read(1) or bim.readline() or rows != sum(counts.values()):
                raise ValueError('PLINK omitted or duplicated records')
    finally:
        for bh, mh in handles.values():
            bh.close(); mh.close()
    return dict(all_hard_calls_verified=True, variants=rows,
                chromosome_counts=counts,
                files=[bound(path) for path in sorted(output_dir.iterdir())])
