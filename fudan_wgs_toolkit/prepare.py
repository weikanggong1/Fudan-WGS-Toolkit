"""Stream PLINK BED/BIM/FAM into validated sparse diploid preparation data.

Only the original BED provides genotypes. Downloaded annotation tables supply
reference alleles, QC, functional weights and gene definitions. Selected BIM
rows require exact annotation matches by default. Explicit source policies can
verify reference alleles in four-component BIM identities and retain missing
non-SNV annotations transparently. A1/A2 are never assumed to mean REF/ALT.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np

from .identity import sample_keys, validate_sample_pairs
from .cache_runtime import store
from .storage_plan import directory_storage, source_storage
from .cache_runtime.portable import (
    FORMAT as METADATA_FORMAT, PortableMetadataReader, _export_field, _sha,
    _value_writer,
)

RAW_ANNOTATION_FORMAT = "raw-wgs-annotations-v1"


def _plink_file(prefix, extension):
    return Path(str(prefix) + extension)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _identity(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def _chromosome(value):
    text = str(value)
    return text[3:] if text.lower().startswith("chr") else text


def _autosome(value):
    text = _chromosome(value)
    if text not in {str(number) for number in range(1, 23)}:
        raise ValueError("chromosome names must be canonical autosomes 1 through 22")
    return text


def _read_fam(path):
    pairs = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for row, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"FAM row {row} must contain six fields")
            pairs.append(fields[:2])
    if not pairs:
        raise ValueError("FAM has no samples")
    return validate_sample_pairs(np.asarray(pairs, dtype=str), where="FAM sample pairs")


def _selected_samples(all_pairs, selection):
    if selection is None:
        return all_pairs, np.arange(len(all_pairs), dtype=np.int64)
    if isinstance(selection, (str, Path)):
        with Path(selection).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {"FID", "IID"}.issubset(reader.fieldnames):
                raise ValueError("sample CSV requires FID and IID columns")
            selection = np.asarray([[row["FID"], row["IID"]] for row in reader], dtype=str)
    pairs = validate_sample_pairs(selection, where="selected sample pairs")
    if not len(pairs):
        raise ValueError("selected sample population is empty")
    lookup = {key: row for row, key in enumerate(sample_keys(all_pairs))}
    try:
        rows = np.asarray([lookup[key] for key in sample_keys(pairs)], dtype=np.int64)
    except KeyError:
        raise ValueError("selected FID/IID pair is absent from FAM") from None
    return pairs, rows


def _prefixes(source_directory, prefixes, chromosomes):
    if isinstance(chromosomes, (str, int, np.integer)):
        chromosomes = [chromosomes]
    requested = None if chromosomes is None else [_autosome(c) for c in chromosomes]
    if requested is not None and (not requested or len(set(requested)) != len(requested)):
        raise ValueError("chromosomes must contain unique chromosome names")
    directory = Path(source_directory).resolve() if source_directory is not None else None
    result = {}
    if prefixes is not None:
        if not isinstance(prefixes, dict) or not prefixes:
            raise ValueError("prefixes must map chromosome names to BED/BIM/FAM prefixes")
        for chromosome, value in prefixes.items():
            path = Path(value)
            if not path.is_absolute():
                if directory is None:
                    raise ValueError("relative prefixes require source_directory")
                path = directory / path
            if path.suffix in (".bed", ".bim", ".fam"):
                path = path.with_suffix("")
            chromosome = _autosome(chromosome)
            if chromosome in result:
                raise ValueError("duplicate normalized chromosome name")
            result[chromosome] = path.resolve()
    else:
        if directory is None:
            raise ValueError("source_directory or prefixes is required")
        for bed in sorted(directory.glob("*.bed")):
            bim = bed.with_suffix(".bim")
            if not bim.exists():
                raise FileNotFoundError("each BED requires a matching BIM")
            with bim.open("r", encoding="utf-8") as stream:
                fields = stream.readline().split()
            if len(fields) != 6:
                raise ValueError("BIM first row must contain six fields")
            chromosome = _autosome(fields[0])
            if chromosome in result:
                raise ValueError("multiple BED prefixes have the same chromosome; pass prefixes explicitly")
            result[chromosome] = bed.with_suffix("").resolve()
    if not result:
        raise ValueError("no BED/BIM/FAM chromosome prefixes found")
    if requested is not None:
        if not set(requested).issubset(result):
            raise ValueError("a requested chromosome has no source prefix")
        result = {chromosome: result[chromosome] for chromosome in requested}
    for prefix in result.values():
        if not all(_plink_file(prefix, extension).is_file() for extension in (".bed", ".bim", ".fam")):
            raise FileNotFoundError("each chromosome needs BED, BIM and FAM files")
    return result


def _annotation_sources(annotation_directory, chromosomes):
    """Resolve an independent raw annotation manifest, never prepared fields."""
    if isinstance(annotation_directory, dict):
        dataset, base = annotation_directory, Path.cwd()
    else:
        path = Path(annotation_directory).resolve()
        if path.is_dir():
            path = path / "annotations.json"
        if not path.is_file():
            raise ValueError("annotation_directory requires annotations.json for original downloaded tables")
        base, dataset = path.parent, json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("format") != RAW_ANNOTATION_FORMAT or dataset.get("schema_version") != 1:
        raise ValueError("annotation input must be a raw-wgs-annotations-v1 manifest; prepared metadata is not accepted")
    if not isinstance(dataset.get("annotation_catalog"), dict) or not isinstance(dataset.get("annotation_names"), list):
        raise ValueError("raw annotations require annotation_catalog and annotation_names")
    entries = {}
    for entry in dataset["chromosomes"]:
        chromosome = _chromosome(entry["name"])
        if chromosome in entries:
            raise ValueError("raw annotation chromosome names must be unique")
        entries[chromosome] = entry
    result = {}
    for chromosome in chromosomes:
        if chromosome not in entries:
            raise ValueError("raw annotation dataset lacks a requested chromosome")
        entry = dict(entries[chromosome])
        mapping = entry.get("column_mapping", dataset.get("column_mapping"))
        if not isinstance(mapping, dict) or not {"chromosome", "position", "reference", "alternate"}.issubset(mapping):
            raise ValueError("column_mapping requires chromosome, position, reference and alternate")
        qc_path = dataset.get("qc_path", "annotation/filter")
        if qc_path not in mapping and entry.get("default_qc", dataset.get("default_qc")) is None:
            raise ValueError("provide a source QC column or an explicit default_qc declaration")
        if not set(dataset["annotation_catalog"].values()).issubset(mapping):
            raise ValueError("raw column_mapping lacks required functional annotation columns")
        variants = entry.get("variants")
        if isinstance(variants, (str, Path)):
            variants = [variants]
        if not isinstance(variants, list) or not variants:
            raise ValueError("each chromosome requires original variant annotation files")
        files = []
        for value in variants:
            path = Path(value)
            path = (path if path.is_absolute() else base / path).resolve()
            if not path.is_file() or not any(str(path).lower().endswith(extension) for extension in
                    (".csv", ".tsv", ".csv.gz", ".tsv.gz", ".parquet", ".tar.gz", ".tgz")):
                raise ValueError("raw annotations must be existing CSV/TSV/gzip/tar CSV or Parquet files")
            files.append(path)
        result[chromosome] = dict(variants=files, column_mapping=mapping,
            delimiter=entry.get("delimiter", dataset.get("delimiter")),
            field_types=dict(dataset.get("field_types", {}), **entry.get("field_types", {})),
            default_qc=entry.get("default_qc", dataset.get("default_qc")),
            source_provenance=dataset.get("source_provenance", {}),
            analysis=dict(annotation_catalog=dataset["annotation_catalog"],
                          annotation_names=dataset["annotation_names"], qc_path=qc_path))
        reference_source = entry.get("reference_allele_source", dataset.get("reference_allele_source", "annotation"))
        allow_missing = entry.get("allow_missing_non_snv", dataset.get("allow_missing_non_snv", False))
        if reference_source not in ("annotation", "bim_variant_id"):
            raise ValueError("reference_allele_source must be annotation or bim_variant_id")
        if type(allow_missing) is not bool:
            raise ValueError("allow_missing_non_snv must be a boolean")
        if allow_missing and reference_source != "bim_variant_id":
            raise ValueError("allow_missing_non_snv requires explicitly verified BIM variant identities")
        if allow_missing and result[chromosome]["default_qc"] is None:
            raise ValueError("allow_missing_non_snv requires an explicit default_qc for unavailable annotations")
        result[chromosome].update(reference_allele_source=reference_source,
                                  allow_missing_non_snv=allow_missing)
        for key in ("gene_reference", "ncrna_reference", "promoter_intervals"):
            value = entry.get(key)
            if value is not None:
                path = Path(value)
                path = (path if path.is_absolute() else base / path).resolve()
                if not path.is_file():
                    raise FileNotFoundError("independent gene/promoter annotation file is missing")
                result[chromosome][key] = path
    return result


def _variant_selection(selection, chromosome, total):
    if isinstance(selection, dict):
        normalized = {}
        for name, rows in selection.items():
            name = _autosome(name)
            if name in normalized:
                raise ValueError("variant selection chromosome names must be unique")
            normalized[name] = rows
        selection = normalized.get(chromosome)
    if selection is None:
        return np.arange(total, dtype=np.int64), False
    values = np.asarray(selection)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError("variant_indices must be a nonempty one-dimensional integer vector")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.any(values >= total) or np.any(values[1:] <= values[:-1]):
        raise ValueError("variant_indices must be increasing unique source BIM rows")
    return values, not (len(values) == total and np.array_equal(values, np.arange(total)))


def _raw_chunks(spec, *, chunk_rows=250000):
    """Stream only requested original annotation columns using C parsers."""
    import pandas as pd
    columns = list(dict.fromkeys(spec["column_mapping"].values()))
    for path in spec["variants"]:
        if str(path).lower().endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ImportError("Parquet annotation input requires the optional pyarrow package") from None
            for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows, columns=columns):
                yield batch.to_pandas().fillna("").astype(str)
        elif str(path).lower().endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r|gz") as archive:
                members = 0
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith((".csv", ".tsv")):
                        continue
                    members += 1
                    delimiter = spec["delimiter"] or ("\t" if member.name.lower().endswith(".tsv") else ",")
                    stream = archive.extractfile(member)
                    # tarfile streaming members do not implement seekable()
                    # on every supported Python version; supply a standard
                    # nonseeking IO layer without unpacking the CSV to disk.
                    class MemberIO(io.RawIOBase):
                        def readable(self):
                            return True
                        def seekable(self):
                            return False
                        def readinto(self, target):
                            value = stream.read(len(target))
                            target[:len(value)] = value
                            return len(value)
                    with stream, io.BufferedReader(MemberIO()) as buffered:
                        yield from pd.read_csv(buffered, sep=delimiter, dtype=str, usecols=columns,
                            keep_default_na=False, chunksize=chunk_rows)
                if not members:
                    raise ValueError("raw annotation archive has no CSV/TSV members")
        else:
            delimiter = spec["delimiter"] or ("\t" if ".tsv" in str(path).lower() else ",")
            yield from pd.read_csv(path, sep=delimiter, dtype=str, usecols=columns,
                keep_default_na=False, chunksize=chunk_rows, compression="infer")


class _RawAnnotationStream:
    """Bounded position merge supporting many extra library records."""
    def __init__(self, spec, chromosome):
        self.spec, self.chromosome = spec, chromosome
        self.iterator, self.current = iter(_raw_chunks(spec,
            chunk_rows=spec.get("_chunk_rows", 250000))), None
        self.last_position, self.done = -1, False
        self.records_read = 0
        self.previous_tail = None

    def _advance(self):
        import pandas as pd
        mapping = self.spec["column_mapping"]
        while True:
            try:
                frame = next(self.iterator)
            except StopIteration:
                self.done, self.current = True, None
                return
            self.records_read += len(frame)
            chromosomes = frame[mapping["chromosome"]].astype(str).str.replace(r"(?i)^chr", "", regex=True)
            frame = frame.loc[chromosomes == self.chromosome].copy()
            if not len(frame):
                continue
            position = frame[mapping["position"]].astype(str)
            if not position.str.fullmatch(r"[0-9]+").all():
                raise ValueError("raw annotation position must be a nonnegative integer")
            frame["_position"] = pd.to_numeric(position, errors="raise").astype(np.int64)
            values = frame["_position"].to_numpy()
            if int(values[0]) < self.last_position or np.any(values[1:] < values[:-1]):
                raise ValueError("original annotation files must be supplied in chromosome/position order")
            self.last_position = int(values[-1])
            frame["_reference"] = frame[mapping["reference"]].astype(str)
            frame["_alternate"] = frame[mapping["alternate"]].astype(str)
            if (frame["_reference"].eq("") | frame["_alternate"].eq("") | frame["_alternate"].str.contains(",", regex=False)).any():
                raise ValueError("raw annotations require one REF and one ALT allele per record")
            direct = frame["_reference"] <= frame["_alternate"]
            frame["_allele_low"] = np.where(direct, frame["_reference"], frame["_alternate"])
            frame["_allele_high"] = np.where(direct, frame["_alternate"], frame["_reference"])
            self.current = frame
            return

    def take(self, positions):
        import pandas as pd
        first, last = int(positions[0]), int(positions[-1])
        pieces = []
        if self.previous_tail is not None:
            repeated = self.previous_tail.loc[(self.previous_tail["_position"] >= first)
                                               & (self.previous_tail["_position"] <= last)]
            if len(repeated):
                pieces.append(repeated)
        while not self.done:
            if self.current is None or not len(self.current):
                self._advance()
                if self.done:
                    break
            values = self.current["_position"].to_numpy()
            if int(values[0]) > last:
                break
            stop = int(np.searchsorted(values, last, side="right"))
            begin = int(np.searchsorted(values, first, side="left"))
            if stop > begin:
                selected = self.current.iloc[begin:stop]
                # Libraries may contain hundreds of millions of extra sites.
                # Retain only positions actually requested in this BIM block,
                # even when a sparse validation subset spans the chromosome.
                selected = selected.loc[selected["_position"].isin(positions)]
                if len(selected):
                    pieces.append(selected)
            self.current = self.current.iloc[stop:]
            if len(self.current):
                break
        if not pieces:
            columns = list(dict.fromkeys(self.spec["column_mapping"].values()))
            columns += ["_position", "_reference", "_alternate", "_allele_low", "_allele_high"]
            result = pd.DataFrame(columns=columns)
        else:
            result = pd.concat(pieces, ignore_index=True)
        self.previous_tail = result.loc[result["_position"] == last].copy()
        return result

    def close(self):
        if hasattr(self.iterator, "close"):
            self.iterator.close()


class _RawPreparedAnnotations(PortableMetadataReader):
    """Read fields made in this preparation directly from original tables."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        raw = (self.directory / "manifest.json").read_bytes()
        if (self.directory / "COMPLETE").read_text() != hashlib.sha256(raw).hexdigest():
            raise ValueError("prepared raw annotations completion differs")
        self.manifest = json.loads(raw)
        if self.manifest.get("format") != RAW_ANNOTATION_FORMAT:
            raise ValueError("only annotations generated from original downloaded tables are accepted")
        self.n_variants, self.n_samples = self.manifest["n_variants"], 0
        self._closed = False
        self._arrays, self._identities, self._validated_offsets = {}, {}, set()
        self._verify_checksums = True
        self._metadata_reads = self._checked_files = 0
        self._metadata_seconds = self._verification_seconds = 0.0
        self._manifest_identity = tuple(_identity(self.directory / "manifest.json")[key]
            for key in ("device", "inode", "size", "mtime_ns", "ctime_ns"))


def _materialize_raw_annotations(prefix, chromosome, spec, directory, total, *,
                                 variant_rows=None, block_size=65536):
    """Validate all source BIM rows and join annotations for the selected axis."""
    import pandas as pd
    selected, subset = _variant_selection(variant_rows, chromosome, total)
    reference_source = spec.get("reference_allele_source", "annotation")
    allow_missing = spec.get("allow_missing_non_snv", False)
    input_binding = dict(bim_sha256=_sha(_plink_file(prefix, ".bim")),
        source_variants=total,
        selected_variant_rows_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        files={str(path): dict(sha256=_sha(path), identity=_identity(path)) for path in spec["variants"]},
        column_mapping=spec["column_mapping"], analysis=spec["analysis"], default_qc=spec["default_qc"],
        reference_allele_source=reference_source, allow_missing_non_snv=allow_missing,
        source_provenance=spec.get("source_provenance", {}),
        reference_files={key: dict(sha256=_sha(spec[key]), identity=_identity(spec[key]))
                         for key in ("gene_reference", "ncrna_reference", "promoter_intervals") if key in spec})
    directory = Path(directory)
    if (directory / "COMPLETE").exists():
        reader = _RawPreparedAnnotations(directory)
        if reader.manifest["input_binding"] != input_binding or reader.n_variants != len(selected):
            reader.close()
            raise ValueError("resumed raw annotation input binding differs")
        orientation = np.load(reader._verify("reference_is_a1.npy"), allow_pickle=False)
        return reader, orientation, reader.manifest["axis_validation"]
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, mode=0o700)
    stream = _RawAnnotationStream(spec, chromosome)
    # Unselected physical source rows have no decoded orientation. An accidental
    # read of those rows is rejected instead of borrowing a reference direction.
    writers, reference_is_a1, digest = {}, np.full(total, 255, dtype=np.uint8), hashlib.sha256()
    counts = dict(variants=len(selected), source_variants=total,
        reference_is_a1=0, reference_is_a2=0, full_variant_axis_checked=True,
        full_annotation_axis_checked=not subset, variant_subset=subset,
        annotation_available=0, missing_non_snv=0, missing_length_difference=0, missing_other=0,
        reference_allele_source=reference_source, allow_missing_non_snv=allow_missing)
    mapping, analysis = spec["column_mapping"], spec["analysis"]
    numeric_fields = {analysis["annotation_catalog"][name] for name in analysis["annotation_names"]}
    last_position, source_row = -1, 0
    started = time.perf_counter()
    def progress(completed=False, **extra):
        store.atomic(directory / "progress.json", _canonical(dict(completed=completed,
            total_variants=len(selected), source_variants=total,
            library_records_read=stream.records_read, seconds=time.perf_counter()-started, **extra)))
    progress(matched_variants=0)
    try:
        with _plink_file(prefix, ".bim").open("rb") as bim:
            def read_row():
                nonlocal last_position, source_row
                line = bim.readline()
                if not line:
                    raise ValueError("BIM has fewer rows than BED variant dimensions")
                digest.update(line)
                fields = line.decode("utf-8").split()
                if len(fields) != 6 or _chromosome(fields[0]) != chromosome:
                    raise ValueError("BIM must have six fields and the requested chromosome")
                try:
                    position = int(fields[3])
                except ValueError:
                    raise ValueError("BIM position must be an integer") from None
                if position < last_position or position < 1:
                    raise ValueError("BIM positions must be positive and ordered")
                last_position = position
                a1, a2 = fields[4], fields[5]
                if a1 == a2 or a1 in ("0", ".") or a2 in ("0", "."):
                    raise ValueError("BIM requires two distinct known alleles")
                reference = alternate = None
                if reference_source == "bim_variant_id":
                    identity = fields[1].split(":")
                    if (len(identity) != 4 or _chromosome(identity[0]) != chromosome
                            or identity[1] != fields[3] or set(identity[2:]) != {a1, a2}):
                        raise ValueError("BIM variant identity must exactly encode chromosome:position:REF:ALT")
                    reference, alternate = identity[2:]
                row = (source_row, position, a1, a2, reference, alternate)
                source_row += 1
                return row
            for begin in range(0, len(selected), block_size):
                stop, rows = min(len(selected), begin+block_size), []
                for target in selected[begin:stop]:
                    while source_row <= int(target):
                        row = read_row()
                    rows.append(row)
                frame = pd.DataFrame(rows, columns=["_source_row", "_position", "_a1", "_a2", "_reference", "_alternate"])
                frame["_row"] = np.arange(len(frame), dtype=np.int64)
                candidates = stream.take(frame["_position"].to_numpy())
                if reference_source == "bim_variant_id":
                    key = ["_position", "_reference", "_alternate"]
                else:
                    frame = frame.drop(columns=["_reference", "_alternate"])
                    direct = frame["_a1"] <= frame["_a2"]
                    frame["_allele_low"] = np.where(direct, frame["_a1"], frame["_a2"])
                    frame["_allele_high"] = np.where(direct, frame["_a2"], frame["_a1"])
                    key = ["_position", "_allele_low", "_allele_high"]
                # Extra library alleles sharing a position cannot make an
                # unrelated requested allele ambiguous.
                candidates = candidates.merge(frame[key].drop_duplicates(), on=key, how="inner", validate="many_to_one")
                duplicated = candidates.duplicated(key, keep=False)
                if duplicated.any():
                    failure = dict(kind="ambiguous_annotation_identity", block_begin=begin,
                        block_end=stop, ambiguous_records=int(duplicated.sum()))
                    progress(failed=True, matched_variants=begin, failure=failure)
                    raise ValueError("raw annotation identity is ambiguous or duplicated")
                candidates = candidates.copy()
                candidates["_annotation_available"] = True
                matched = frame.merge(candidates, how="left", on=key, validate="many_to_one", sort=False).sort_values("_row")
                available = matched["_annotation_available"].eq(True)
                missing = ~available
                if missing.any():
                    unknown = matched.loc[missing]
                    length_a1, length_a2 = unknown["_a1"].str.len(), unknown["_a2"].str.len()
                    single = (length_a1 == 1) & (length_a2 == 1)
                    length_difference = length_a1 != length_a2
                    failure = dict(kind="missing_exact_annotation", block_begin=begin,
                        block_end=stop, checked_variants=len(matched),
                        matched_variants=int(available.sum()), missing_variants=int(missing.sum()),
                        missing_snv=int(single.sum()),
                        missing_length_difference=int(length_difference.sum()),
                        missing_other=int((~single & ~length_difference).sum()))
                    if not allow_missing or single.any():
                        progress(failed=True, matched_variants=begin, failure=failure)
                        raise ValueError("a source BIM variant has no exact CHR/POS/REF/ALT annotation match")
                    counts["missing_non_snv"] += int(missing.sum())
                    counts["missing_length_difference"] += int(length_difference.sum())
                    counts["missing_other"] += int((~single & ~length_difference).sum())
                orientation = matched["_a1"].to_numpy() == matched["_reference"].to_numpy()
                reference_is_a1[selected[begin:stop]] = orientation.astype(np.uint8)
                counts["reference_is_a1"] += int(np.count_nonzero(orientation))
                counts["reference_is_a2"] += int(len(orientation)-np.count_nonzero(orientation))
                counts["annotation_available"] += int(available.sum())
                values = {"position": matched["_position"].to_numpy(dtype=np.int64),
                    "chromosome": np.full(len(matched), chromosome, dtype=object),
                    "variant.id": matched["_source_row"].to_numpy(dtype=np.int64)+1,
                    "allele": (matched["_reference"].astype(str)+","+matched["_alternate"].astype(str)).to_numpy(dtype=object),
                    "annotation_available": available.to_numpy(dtype=bool)}
                for path, column in mapping.items():
                    if path in ("chromosome", "position", "reference", "alternate"):
                        continue
                    source = matched[column]
                    if path in numeric_fields or np.dtype(spec["field_types"].get(path, "O")).kind in "iuf":
                        clean = source.replace({"": np.nan, ".": np.nan, "NA": np.nan, "NaN": np.nan})
                        values[path] = pd.to_numeric(clean, errors="raise").to_numpy(dtype=np.float64)
                    else:
                        values[path] = source.fillna("").to_numpy(dtype=object)
                if analysis["qc_path"] not in values:
                    values[analysis["qc_path"]] = np.full(len(matched), str(spec["default_qc"]), dtype=object)
                elif missing.any() and allow_missing:
                    values[analysis["qc_path"]][missing.to_numpy()] = str(spec["default_qc"])
                for path, value in values.items():
                    if path not in writers:
                        writers[path] = _value_writer(directory, "field_%03d" % len(writers), value)
                    writers[path].append(value)
                progress(matched_variants=stop, annotation_available=counts["annotation_available"],
                         missing_non_snv=counts["missing_non_snv"])
            # Even a small selected annotation axis proves the complete source
            # BIM structure, order and bytes, without scanning unneeded tables.
            while source_row < total:
                read_row()
            if bim.read():
                raise ValueError("BIM has more rows than BED variant dimensions")
        counts["bim_sha256"], counts["library_records_read"] = digest.hexdigest(), stream.records_read
        if counts["bim_sha256"] != input_binding["bim_sha256"]:
            raise RuntimeError("source BIM changed while annotations were joined")
        for path in spec["variants"]:
            if _identity(path) != input_binding["files"][str(path)]["identity"]:
                raise RuntimeError("original annotation source changed while annotations were joined")
        fields = {path: writer.finish() for path, writer in writers.items()}
        progress(completed=True, matched_variants=len(selected), annotation_available=counts["annotation_available"],
                 missing_non_snv=counts["missing_non_snv"])
        np.save(directory / "reference_is_a1.npy", reference_is_a1, allow_pickle=False)
        np.save(directory / "source_variant_rows.npy", selected, allow_pickle=False)
        files = {path.name: dict(sha256=_sha(path), size=path.stat().st_size)
                 for path in directory.iterdir() if path.is_file()}
        manifest = dict(schema_version=1, format=RAW_ANNOTATION_FORMAT, n_variants=len(selected),
                        source_variants=total, selected_variant_axis=True, fields=fields, files=files,
                        analysis=analysis, input_binding=input_binding, axis_validation=counts)
        raw = _canonical(manifest)
        store.atomic(directory / "manifest.json", raw)
        store.atomic(directory / "COMPLETE", hashlib.sha256(raw).hexdigest().encode())
        return _RawPreparedAnnotations(directory), reference_is_a1, counts
    finally:
        stream.close()
        for writer in writers.values():
            writer.close()


def _read_bed_frame(stream, variant_rows, sample_rows, n_source_samples,
                    reference_is_a1, *, cpu_threads):
    stride = (n_source_samples + 3) // 4
    variant_rows = np.asarray(variant_rows, dtype=np.int64)
    if len(variant_rows) > 1 and np.all(np.diff(variant_rows) == 1):
        payload = os.pread(stream.fileno(), len(variant_rows) * stride,
                           3 + int(variant_rows[0]) * stride)
        if len(payload) != len(variant_rows) * stride:
            raise ValueError("short BED frame read")
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(len(variant_rows), stride)
    else:
        packed = np.empty((len(variant_rows), stride), dtype=np.uint8)
        for local, variant in enumerate(variant_rows):
            payload = os.pread(stream.fileno(), stride, 3 + int(variant) * stride)
            if len(payload) != stride:
                raise ValueError("short BED variant read")
            packed[local] = np.frombuffer(payload, dtype=np.uint8)
    # Every unused high two-bit sample slot must be zero in the standard BED.
    if n_source_samples % 4:
        mask = 255 ^ ((1 << (2 * (n_source_samples % 4))) - 1)
        if np.any(packed[:, -1] & np.uint8(mask)):
            raise ValueError("BED contains nonzero sample padding bits")
    states = np.empty((len(variant_rows), len(sample_rows)), dtype=np.uint8)
    byte_rows = sample_rows // 4
    shifts = ((sample_rows % 4) * 2).astype(np.uint8)
    maps = np.asarray([[2, 3, 1, 0], [0, 3, 1, 2]], dtype=np.uint8)
    selected_orientation = reference_is_a1[variant_rows]
    if np.any((selected_orientation != 0) & (selected_orientation != 1)):
        raise ValueError("BED variant has no validated reference direction")
    orientation = selected_orientation.astype(np.int64)

    def decode(begin, stop):
        code = packed[begin:stop, byte_rows]
        np.right_shift(code, shifts, out=code)
        np.bitwise_and(code, np.uint8(3), out=code)
        states[begin:stop] = maps[orientation[begin:stop, None], code]

    workers = min(cpu_threads, len(variant_rows))
    if workers <= 1:
        decode(0, len(variant_rows))
    else:
        cuts = np.linspace(0, len(variant_rows), workers + 1, dtype=np.int64)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(decode, int(cuts[i]), int(cuts[i + 1])) for i in range(workers)]
            for future in futures:
                future.result()
    reference_count = 2 * np.count_nonzero(states == 0, axis=1) + np.count_nonzero(states == 1, axis=1)
    called_count = 2 * (len(sample_rows) - np.count_nonzero(states == 3, axis=1))
    counts = dict(reference_alleles=reference_count.astype(np.int64),
                  called_alleles=called_count.astype(np.int64),
                  half_missing_samples=np.zeros(len(variant_rows), dtype=np.int64))
    return states, counts


def _field_files(spec):
    result = set()
    for key, value in spec.items():
        if key == "file":
            result.add(value)
        elif isinstance(value, dict):
            result.update(_field_files(value))
    return result


def _coordinate(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("gene/promoter coordinates must be integers") from None
    if not number.is_finite() or number != int(number) or number < 1:
        raise ValueError("gene/promoter coordinates must be positive integers")
    return int(number)


def _reference_rows(path, required, *, delimiter=","):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("independent gene/promoter reference lacks required CSV/TSV columns")
        yield from reader


def _write_gene_catalogs(spec, chromosome, output):
    """Build ordered gene jobs only from independent original reference CSVs."""
    directory = Path(output) / "catalogs"
    directory.mkdir(exist_ok=True, mode=0o700)
    entries, genes, ncrna = {}, [], []
    source = spec.get("gene_reference")
    if source is not None:
        required = {"hgnc_symbol", "chromosome_name", "start_position", "end_position"}
        for row in _reference_rows(source, required):
            if _chromosome(row["chromosome_name"]) != chromosome:
                continue
            name = row["hgnc_symbol"]
            start, end = _coordinate(row["start_position"]), _coordinate(row["end_position"])
            if not name or start > end:
                raise ValueError("gene names must be nonempty with start <= end")
            genes.append((name, start, end))
    source = spec.get("ncrna_reference")
    if source is not None:
        for row in _reference_rows(source, {"chr", "ncRNA"}):
            if _chromosome(row["chr"]) != chromosome:
                continue
            if not row["ncRNA"]:
                raise ValueError("noncoding RNA reference names must be nonempty")
            ncrna.append(row["ncRNA"])
    if genes or ncrna:
        jobs = [dict(kind="coding", arguments=dict(gene_name=name, start=start, end=end,
                    category="all_categories", include_ptv=False)) for name, start, end in genes]
        jobs += [dict(kind="noncoding", arguments=dict(gene_name=name, category="all_categories"))
                 for name, _, _ in genes]
        jobs += [dict(kind="ncrna", arguments=dict(gene_name=name)) for name in ncrna]
        path = directory / ("chr" + chromosome + ".genes.json")
        provenance = {key: dict(sha256=_sha(spec[key]), size=spec[key].stat().st_size)
                      for key in ("gene_reference", "ncrna_reference") if key in spec}
        store.atomic(path, _canonical(dict(jobs=jobs, source_format="independent-reference-csv",
                                           source_files=provenance)))
        entries["gene_catalog"] = str(path.relative_to(output))
    source = spec.get("promoter_intervals")
    if source is not None:
        intervals = []
        for row in _reference_rows(source, {"chromosome", "start", "end"}, delimiter="\t"):
            if _chromosome(row["chromosome"]) != chromosome:
                continue
            start, end = _coordinate(row["start"]), _coordinate(row["end"])
            if start > end:
                raise ValueError("promoter interval start must not exceed end")
            intervals.append([chromosome, start, end])
        path = directory / ("chr" + chromosome + ".promoters.json")
        store.atomic(path, _canonical(intervals))
        entries["promoter_intervals"] = str(path.relative_to(output))
    return entries


def _write_metadata(annotations, container_directory, pairs, source_rows,
                    variant_rows, variant_subset, *, hardlink_annotations, n_source_samples):
    directory = Path(container_directory) / "metadata"
    if directory.exists():
        # Resume can reach this branch after metadata was committed but before
        # the root dataset was published. Verify every immutable metadata file,
        # including fields that analysis would otherwise read only lazily.
        with PortableMetadataReader(directory, container_directory) as reader:
            existing = reader.manifest
            for name in existing["files"]:
                reader._verify(name)
            if (reader.n_variants != len(variant_rows)
                    or existing["n_source_samples"] != n_source_samples
                    or not np.array_equal(reader._array(existing["sample_pairs"]), pairs)
                    or not np.array_equal(reader._array(existing["source_sample_rows"]), source_rows)
                    or not np.array_equal(reader._array(existing["source_variant_rows"]), variant_rows)
                    or existing["analysis"] != annotations.manifest["analysis"]
                    or existing["preparation"]["original_annotation_input"] != annotations.manifest["input_binding"]):
                raise ValueError("existing metadata belongs to another prepared source/sample/annotation axis")
            proof = existing["preparation"].get("raw_annotation_manifest")
            if proof is not None:
                container = json.loads((Path(container_directory) / "manifest.json").read_text())
                if _sha(reader._verify(proof)) != container["binding"]["raw_annotation_manifest_sha256"]:
                    raise ValueError("retained annotation preflight manifest differs from genotype binding")
            return existing
    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    np.save(staging / "sample_pairs.npy", pairs, allow_pickle=False)
    np.save(staging / "sample_ids.npy", sample_keys(pairs), allow_pickle=False)
    np.save(staging / "source_sample_rows.npy", source_rows, allow_pickle=False)
    np.save(staging / "source_variant_rows.npy", variant_rows, allow_pickle=False)
    try:
        if annotations.manifest.get("selected_variant_axis"):
            if not np.array_equal(np.load(annotations._verify("source_variant_rows.npy"), allow_pickle=False), variant_rows):
                raise ValueError("prepared annotation variant selection differs")
        if variant_subset and not annotations.manifest.get("selected_variant_axis"):
            class SelectedAnnotations:
                n_variants = len(variant_rows)
                def read_field(self, path, indices):
                    return annotations.read_field(path, variant_rows[indices])
            fields = {path: _export_field(SelectedAnnotations(), path, staging,
                                          "field_%03d" % index, 65536)
                      for index, path in enumerate(annotations.manifest["fields"])}
        else:
            fields = annotations.manifest["fields"]
            for name in sorted(set().union(*(_field_files(spec) for spec in fields.values()))):
                source = annotations._verify(name)
                target = staging / name
                if hardlink_annotations:
                    try:
                        os.link(source, target)
                        # Our own link increments inode ctime. Confirm exact
                        # bytes against the proven source checksum before
                        # updating that reader's immutable identity witness.
                        if _sha(target) != annotations.manifest["files"][name]["sha256"]:
                            raise ValueError("raw annotation hardlink bytes differ")
                        annotations._identities[name] = tuple(_identity(source)[key]
                            for key in ("device", "inode", "size", "mtime_ns", "ctime_ns"))
                    except OSError:
                        shutil.copyfile(source, target)
                else:
                    shutil.copyfile(source, target)
        container = json.loads((Path(container_directory) / "manifest.json").read_text())
        # Retain the small preflight witness while releasing its large field
        # staging after root commitment. Its exact bytes close the provenance
        # hash chain named by the genotype container's source binding.
        annotation_manifest = (annotations.directory / "manifest.json").read_bytes()
        if hashlib.sha256(annotation_manifest).hexdigest() != container["binding"]["raw_annotation_manifest_sha256"]:
            raise ValueError("annotation preflight manifest differs from genotype binding")
        (staging / "annotation.source.json").write_bytes(annotation_manifest)
        files = {path.name: dict(sha256=_sha(path), size=path.stat().st_size)
                 for path in staging.iterdir() if path.is_file()}
        metadata = dict(schema_version=1, format=METADATA_FORMAT,
            n_samples=len(pairs), n_source_samples=n_source_samples,
            source_sample_rows="source_sample_rows.npy", source_variant_rows="source_variant_rows.npy",
            n_variants=len(variant_rows), sample_ids="sample_ids.npy", sample_pairs="sample_pairs.npy",
            sample_identifier_format="fid_iid_json",
            genotype_manifest_sha256=_sha(Path(container_directory) / "manifest.json"),
            genotype_sample_axis_sha256=container["sample_sha256"], fields=fields, files=files,
            analysis=annotations.manifest["analysis"],
            annotation_coverage=dict(n_variants=len(variant_rows),
                matched_variants=annotations.manifest["axis_validation"]["annotation_available"],
                missing_snv=0,
                missing_non_snv=annotations.manifest["axis_validation"]["missing_non_snv"],
                allow_missing_non_snv=annotations.manifest["axis_validation"]["allow_missing_non_snv"],
                reference_allele_source=annotations.manifest["axis_validation"]["reference_allele_source"],
                available_field="annotation_available"),
            preparation=dict(source_format="PLINK-BED-BIM-FAM", annotation_source_format=RAW_ANNOTATION_FORMAT,
                             original_annotation_input=annotations.manifest["input_binding"], variant_subset=variant_subset,
                             half_missing_source_supported=False, hardlink_annotations=hardlink_annotations,
                             raw_annotation_manifest="annotation.source.json"))
        raw = _canonical(metadata)
        (staging / "manifest.json").write_bytes(raw)
        (staging / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
        os.replace(staging, directory)
        return metadata
    except BaseException:
        shutil.rmtree(staging)
        raise



_PROCESS_START_GATE = None


def _worker_initialize(cpu_threads, gate):
    """Spawned processes receive a share of the caller's total CPU budget."""
    global _PROCESS_START_GATE
    _PROCESS_START_GATE = gate
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[name] = str(cpu_threads)
    # The package imports torch for association, but preparation performs no
    # CUDA initialization. Bound its otherwise idle native thread pools too.
    import torch
    torch.set_num_threads(cpu_threads)


def _check_sources(context):
    prefix = Path(context["prefix"])
    actual = {extension: _identity(_plink_file(prefix, extension))
              for extension in context["source_before"]}
    if actual != context["source_before"]:
        raise RuntimeError("PLINK source changed during preparation")
    for name, expected in context.get("descriptor_files_sha256", {}).items():
        if _sha(context[name + "_file"]) != expected:
            raise RuntimeError("preflight sample or variant descriptor changed during preparation")
    binding = context.get("annotation_input_binding", {})
    for name, proof in binding.get("files", {}).items():
        if _identity(name) != proof["identity"]:
            raise RuntimeError("original annotation source changed during preparation")
    for key, proof in binding.get("reference_files", {}).items():
        if _identity(context["annotation_spec"][key]) != proof["identity"]:
            raise RuntimeError("independent gene/promoter reference changed during preparation")


def _source_storage_paths(contexts):
    """Original inputs remain external; count shared tables only once."""
    paths = set()
    for context in contexts:
        for extension in context["source_before"]:
            paths.add(_plink_file(context["prefix"], extension))
        paths.update(Path(name) for name in context.get("annotation_input_binding", {}).get("files", {}))
        for key in context.get("annotation_input_binding", {}).get("reference_files", {}):
            paths.add(Path(context["annotation_spec"][key]))
    return sorted(paths)


def _preparation_storage(output, contexts, *, completed, before_cleanup=None,
                         cleanup=None):
    """Report measured file bytes separately from source and cache admission.

    The sidecar excludes itself, so its byte counts have a precise stable
    scope. Phase-boundary measurements are not claimed to be a disk high-water
    mark: atomic temporary files can coexist between those measurements.
    """
    output = Path(output)
    def owned_directory(path):
        if path.is_symlink():
            raise ValueError("prepared component directory must not be a symlink")
        return directory_storage(path) if path.exists() else source_storage([])
    retained = directory_storage(output, exclude=("storage.report.json",))
    components = dict(
        genotype_cache=source_storage([path for context in contexts
            for path in (output / ("chr" + context["chromosome"])).glob("*") if path.is_file()]),
        metadata=source_storage([path for context in contexts
            if (path := output / ("chr" + context["chromosome"]) / "metadata").exists()]),
        catalogs=owned_directory(output / "catalogs"),
        raw_annotation_staging=owned_directory(output / ".raw_annotations"),
        sample_and_variant_staging=owned_directory(output / ".preflight"))
    caps = [dict(chromosome=context["chromosome"],
                 retained_bed_bytes=context["source_before"][".bed"]["size"],
                 genotype_cache_limit_bytes=max(int(context["source_before"][".bed"]["size"] * .10), 2**20))
            for context in contexts]
    boundaries = [retained] + ([] if before_cleanup is None else [before_cleanup])
    basis = ("unique_allocated_bytes" if all(item["unique_allocated_bytes"] is not None
                                           for item in boundaries) else "unique_logical_bytes")
    report = dict(schema_version=1, format="wgs-preparation-storage-v1",
        completed=bool(completed), unit="bytes", measurement="file_stat",
        scope="prepared output excluding storage.report.json; symlinks are not traversed",
        original_inputs_retained=True, source=source_storage(_source_storage_paths(contexts)),
        prepared=retained, components=components,
        genotype_cache_admission=dict(kind="retained_bed_fraction", fraction=.10,
            minimum_bytes_per_chromosome=2**20, chromosomes=caps,
            includes_metadata=False, includes_retained_original_inputs=False),
        phase_boundary_before_cleanup=before_cleanup,
        peak=dict(actual_bytes=None, observed_phase_boundary_at_least_bytes=max(item[basis] for item in boundaries),
            basis=basis,
            reason="phase-boundary file measurements do not observe every atomic temporary file"),
        cleanup=cleanup or dict(completed=False, removed=[], errors=[]))
    store.atomic(output / "storage.report.json", _canonical(report))
    return report


def _cleanup_committed_staging(output, dataset_bytes):
    """Remove only private staging after the exact root commit is durable."""
    output = Path(output)
    if ((output / "dataset.json").read_bytes() != dataset_bytes or
            (output / "COMPLETE").read_text() != hashlib.sha256(dataset_bytes).hexdigest()):
        raise ValueError("root dataset commitment differs before staging cleanup")
    committed_root = output.resolve(strict=True)
    removed, errors = [], []
    for name in (".preflight", ".raw_annotations"):
        path = output / name
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            errors.append(dict(directory=name, error="symlink_staging_retained"))
            continue
        if not path.resolve(strict=True).is_relative_to(committed_root):
            raise ValueError("preparation staging escapes the committed output directory")
        try:
            shutil.rmtree(path)
            removed.append(name)
        except OSError as error:
            errors.append(dict(directory=name, error=type(error).__name__, errno=error.errno))
    return dict(completed=not errors, removed=removed, errors=errors)


def _source_descriptor(chromosome, prefix, specification, output, sample_selection,
                       variant_selection, chunk_size):
    """Check common sample order early; exchange only paths with processes."""
    before = {extension: _identity(_plink_file(prefix, extension))
              for extension in (".bed", ".bim", ".fam")}
    all_pairs = _read_fam(_plink_file(prefix, ".fam"))
    pairs, rows = _selected_samples(all_pairs, sample_selection)
    stride = (len(all_pairs) + 3) // 4
    with _plink_file(prefix, ".bed").open("rb") as stream:
        if stream.read(3) != b"\x6c\x1b\x01":
            raise ValueError("BED must use the SNP-major 6c1b01 header")
    if (before[".bed"]["size"] - 3) % stride:
        raise ValueError("BED byte length differs from complete FAM sample strides")
    total = (before[".bed"]["size"] - 3) // stride
    if not total:
        raise ValueError("BED contains no variants")
    selected, subset = _variant_selection(variant_selection, chromosome, total)
    directory = Path(output) / ".preflight" / ("chr" + chromosome)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    arrays = {"pairs": pairs, "sample_rows": rows, "variant_rows": selected}
    paths = {}
    hashes = {}
    for name, value in arrays.items():
        path = directory / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        paths[name + "_file"] = str(path)
        hashes[name] = _sha(path)
    # Decoding, sparse validation and axes coexist in a process. Annotation
    # parsing receives a separate bounded chunk allowance from the scheduler.
    frame_bytes = 16 * min(chunk_size, total) * len(pairs) + stride * chunk_size
    axis_bytes = 24 * total + pairs.nbytes + rows.nbytes
    fields = len(set(specification["column_mapping"].values())) + 8
    join_bytes = min(65536, len(selected)) * fields * 256
    baseline = axis_bytes + 8 * 2**20
    context = dict(chromosome=chromosome, prefix=str(prefix), output=str(output),
        source_before=before, annotation_spec=specification,
        n_source_samples=len(all_pairs), n_source_variants=total,
        n_samples=len(pairs), n_variants=len(selected), variant_subset=subset,
        conversion_workspace_bytes=baseline + frame_bytes,
        preflight_workspace_base_bytes=baseline + join_bytes,
        annotation_row_workspace_bytes=fields * 256,
        descriptor_files_sha256=hashes, **paths)
    _check_sources(context)
    return context, pairs


def _preparation_schedule(contexts, cpu_threads, memory_limit_gib, *, max_frames):
    limit = int(memory_limit_gib * 2**30)
    minimum = max(max(context["conversion_workspace_bytes"],
                      context["preflight_workspace_base_bytes"] +
                      context["annotation_row_workspace_bytes"]) for context in contexts)
    if minimum > limit:
        raise MemoryError("BED conversion workspace exceeds memory_limit_gib; reduce chunk_size")
    requested = min(len(contexts), cpu_threads)
    workers = min(requested, max(1, limit // minimum))
    per_worker_limit = limit // workers
    for context in contexts:
        context["annotation_spec"] = dict(context["annotation_spec"])
        rows = min(250000, max(1, (per_worker_limit -
            context["preflight_workspace_base_bytes"]) // context["annotation_row_workspace_bytes"]))
        context["annotation_spec"]["_chunk_rows"] = int(rows)
        context["preflight_workspace_bytes"] = (context["preflight_workspace_base_bytes"] +
            rows * context["annotation_row_workspace_bytes"])
    conversion_workers = 1 if max_frames is not None else workers
    reasons = []
    if workers < requested:
        reasons.append("aggregate_workspace_admission")
    if len(contexts) == 1:
        reasons.append("single_chromosome_threaded_decode")
    if cpu_threads == 1:
        reasons.append("single_cpu_budget")
    if max_frames is not None:
        reasons.append("global_frame_checkpoint_serial_conversion")
    return dict(requested_cpu_threads=cpu_threads, requested_chromosome_workers=requested,
        budget_is_estimate=True,
        preflight_workers=workers, conversion_workers=conversion_workers,
        preflight_cpu_threads_per_worker=max(1, cpu_threads // workers),
        conversion_cpu_threads_per_worker=max(1, cpu_threads // conversion_workers),
        process_start_method="spawn" if workers > 1 else "none",
        memory_limit_bytes=limit, per_preflight_worker_limit_bytes=per_worker_limit,
        preflight_workspace_aggregate_bound_bytes=workers * max(c["preflight_workspace_bytes"] for c in contexts),
        conversion_workspace_aggregate_bound_bytes=conversion_workers * max(c["conversion_workspace_bytes"] for c in contexts),
        limiting_reasons=reasons)


def _preflight_chromosome(context):
    """Resolve independent tables before any chromosome genotype is decoded."""
    started = time.perf_counter()
    _check_sources(context)
    chromosome, prefix = context["chromosome"], Path(context["prefix"])
    specification = context["annotation_spec"]
    references_before = {key: dict(sha256=_sha(specification[key]), identity=_identity(specification[key]))
        for key in ("gene_reference", "ncrna_reference", "promoter_intervals") if key in specification}
    catalogs = _write_gene_catalogs(specification, chromosome, Path(context["output"]))
    selected = np.load(context["variant_rows_file"], mmap_mode="r", allow_pickle=False)
    annotations, orientation, validation = _materialize_raw_annotations(prefix, chromosome,
        specification, Path(context["output"]) / ".raw_annotations" / ("chr" + chromosome),
        context["n_source_variants"], variant_rows=selected)
    try:
        if annotations.manifest["input_binding"]["reference_files"] != references_before:
            raise RuntimeError("independent gene/promoter reference changed during preflight")
        pairs = np.load(context["pairs_file"], mmap_mode="r", allow_pickle=False)
        result = dict(context, catalogs=catalogs, axis_validation=validation,
            analysis=annotations.manifest["analysis"],
            annotation_input_binding=annotations.manifest["input_binding"],
            raw_annotation_directory=str(annotations.directory),
            binding=dict(schema_version=1, source_format="PLINK-BED-BIM-FAM",
                source_identity=context["source_before"], fam_sha256=_sha(_plink_file(prefix, ".fam")),
                bim_sha256=validation["bim_sha256"],
                raw_annotation_manifest_sha256=_sha(annotations.directory / "manifest.json"),
                sample_pairs_sha256=hashlib.sha256(_canonical(pairs.tolist())).hexdigest(),
                selected_variant_rows_sha256=hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                source_samples=context["n_source_samples"], source_variants=context["n_source_variants"]),
            preflight_pid=os.getpid(), preflight_seconds=time.perf_counter()-started)
        _check_sources(result)
        return result
    finally:
        annotations.close()


def _convert_chromosome(context, *, cpu_threads, hardlink_annotations, chunk_size,
                        max_frames=None):
    """Own one chromosome Writer and reopen all arrays locally by filename."""
    _check_sources(context)
    prefix = Path(context["prefix"])
    directory = Path(context["output"]) / ("chr" + context["chromosome"])
    selected = np.load(context["variant_rows_file"], mmap_mode="r", allow_pickle=False)
    rows = np.load(context["sample_rows_file"], mmap_mode="r", allow_pickle=False)
    pairs = np.load(context["pairs_file"], mmap_mode="r", allow_pickle=False)
    converted_started = time.perf_counter()
    cache_complete = (directory / "COMPLETE").exists()
    written_frames = 0
    with _RawPreparedAnnotations(context["raw_annotation_directory"]) as annotations:
        if _sha(annotations.directory / "manifest.json") != context["binding"]["raw_annotation_manifest_sha256"]:
            raise RuntimeError("resolved raw annotations changed after preflight")
        orientation = np.load(annotations._verify("reference_is_a1.npy"), mmap_mode="r", allow_pickle=False)
        if cache_complete:
            existing = store.Container(directory, expected_source_binding=context["binding"], expected_samples=rows)
            if existing.manifest["m"] != len(selected):
                raise ValueError("completed chromosome has another variant selection")
            existing.verify_streams()
        else:
            pending_manifest = directory / "manifest.pending.json"
            # Existing checkpoints keep their original journal contract. New
            # real-source conversions use the bounded append-only journal,
            # without relaxing the retained-BED storage fraction.
            pending_spec = json.loads(pending_manifest.read_text()) if pending_manifest.exists() else None
            streaming = pending_spec is None or pending_spec.get("journal_format") == "jsonl-v1"
            sample_encoding = "row-delta" if pending_spec is None else pending_spec.get("sample_encoding", "absolute")
            writer = store.Writer(directory, context["binding"], rows, len(selected),
                source_bytes=context["source_before"][".bed"]["size"],
                minimum_storage_bytes=2**20, chunk=chunk_size, streaming=streaming,
                sample_encoding=sample_encoding)
            with _plink_file(prefix, ".bed").open("rb") as stream:
                while writer.next_start < len(selected):
                    if max_frames is not None and written_frames >= max_frames:
                        return dict(completed=False, committed_frames=written_frames)
                    start, stop = writer.next_start, min(writer.next_start + chunk_size, len(selected))
                    states, counts = _read_bed_frame(stream, selected[start:stop], rows,
                        context["n_source_samples"], orientation, cpu_threads=cpu_threads)
                    writer.append(states, counts)
                    written_frames += 1
                    del states, counts
            _check_sources(context)
            annotations._verify("reference_is_a1.npy")
            writer.finish()
        _write_metadata(annotations, directory, pairs, rows, selected, context["variant_subset"],
            hardlink_annotations=hardlink_annotations, n_source_samples=context["n_source_samples"])
    _check_sources(context)
    entry = dict(name=context["chromosome"], container_directory=directory.name,
                 metadata_directory=directory.name + "/metadata", **context["catalogs"])
    metrics = dict(chromosome=context["chromosome"], n_samples=len(rows), n_variants=len(selected),
        source_n_samples=context["n_source_samples"], source_n_variants=context["n_source_variants"],
        variant_subset=context["variant_subset"], bed_conversion_seconds=time.perf_counter()-converted_started,
        axis_validation=context["axis_validation"], resumed_completed=cache_complete,
        preflight_pid=context["preflight_pid"], conversion_pid=os.getpid(),
        preflight_seconds=context["preflight_seconds"], decode_cpu_threads=cpu_threads,
        effective_max_decode_threads=min(cpu_threads, chunk_size, len(selected)),
        native_thread_environment={name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        annotation_chunk_rows=context["annotation_spec"]["_chunk_rows"],
        estimated_conversion_workspace_bytes=context["conversion_workspace_bytes"],
        estimated_preflight_workspace_bytes=context["preflight_workspace_bytes"])
    return dict(completed=True, entry=entry, metrics=metrics, committed_frames=written_frames)


def _preflight_group(contexts):
    _PROCESS_START_GATE.wait(timeout=120)
    return [_preflight_chromosome(context) for context in contexts]


def _conversion_group(arguments):
    contexts, options = arguments
    _PROCESS_START_GATE.wait(timeout=120)
    return [_convert_chromosome(context, **options) for context in contexts]


def _process_groups(function, contexts, workers, cpu_threads, *, options=None):
    """Use spawn and path descriptors; a start gate gives each worker a group."""
    context = multiprocessing.get_context("spawn")
    groups = [contexts[index::workers] for index in range(workers)]
    arguments = groups if options is None else [(group, options) for group in groups]
    gate = context.Barrier(workers)
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        # Spawn imports NumPy before executing its initializer. Set this share
        # before process creation so native libraries inherit the same bound.
        for name in names:
            os.environ[name] = str(cpu_threads)
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=_worker_initialize, initargs=(cpu_threads, gate)) as executor:
            results = list(executor.map(function, arguments))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    flat = [result for group in results for result in group]
    # Static balanced process groups must not change requested chromosome order.
    if options is None:
        lookup = {result["chromosome"]: result for result in flat}
        return [lookup[original["chromosome"]] for original in contexts]
    lookup = {result["entry"]["name"]: result for result in flat}
    return [lookup[original["chromosome"]] for original in contexts]

def prepare_WGS_data(source_directory=None, output_directory=None, *, prefixes=None,
                     chromosomes=None, annotation_directory=None, sample_pairs=None,
                     variant_indices=None, memory_limit_gib=8, cpu_threads=8,
                     hardlink_annotations=True, resume=False, max_frames=None,
                     chunk_size=256):
    """Prepare chromosome data before :func:`run_WGS_all_binanry`.

    ``prefixes`` maps chromosome names to source BED/BIM/FAM prefixes; otherwise
    BED files are discovered in ``source_directory``. ``annotation_directory``
    supplies an independent ``annotations.json`` manifest with original
    downloaded annotation tables, column mappings and gene definitions.
    Previously prepared metadata is not an annotation input. FID/IID are
    retained exactly. Optional
    ``sample_pairs`` selects/reorders those pairs; optional ``variant_indices``
    selects increasing zero-based BIM rows and explicitly marks a subset run.
    The manifest may explicitly declare ``reference_allele_source`` as
    ``bim_variant_id`` for identities encoding chromosome:position:REF:ALT.
    The default requires complete annotation coverage. An explicit
    ``allow_missing_non_snv=True`` declaration retains unannotated non-SNV
    genotypes with NaN numeric weights, empty text annotations and an
    ``annotation_available=False`` flag; SNV annotation is always required.

    ``cpu_threads`` is the total CPU budget. Multiple chromosomes use spawned
    processes by default; each process receives a share of the decoding threads.
    One chromosome uses the budget for NumPy BED decoding. Standalone Python
    scripts must place multi-chromosome calls under an ``if __name__ == "__main__"``
    guard. ``cpu_threads=1`` keeps both stages serial.
    ``memory_limit_gib`` is an estimated host workspace scheduling budget,
    not a hard RSS cap. String tables, sparse encoding scratch and interpreter
    memory may exceed the estimate; the report identifies it as an estimate.
    ``max_frames`` serializes conversion to enforce its exact global frame limit,
    stops at a resumable checkpoint and never publishes a dataset
    manifest until all selected chromosomes are complete. ``resume=True`` checks
    the exact preparation/source/sample binding before continuing.

    New caches use an append-only frame journal and a mapped final index.
    The ordinary genotype-cache limit remains 10% of retained BED bytes,
    with a 1 MiB minimum per chromosome; it does not include annotation
    metadata or original input files. ``storage.report.json`` distinguishes
    logical bytes, allocated bytes, unique hardlinked storage, retained
    sources and temporary staging. Staging is removed only after the final
    root dataset and completion marker have been written and verified.

    BED supplies complete diploid calls or fully missing calls only. Partial
    allele calls cannot be reconstructed from this format. The six-state format
    remains unchanged and every written CSR frame is checked by roundtrip.
    """
    if output_directory is None or annotation_directory is None:
        raise ValueError("output_directory and annotation_directory are required")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if isinstance(memory_limit_gib, bool) or not isinstance(memory_limit_gib, (int, float)) or not math.isfinite(memory_limit_gib) or memory_limit_gib <= 0:
        raise ValueError("memory_limit_gib must be positive and finite")
    if type(chunk_size) is not int or not 1 <= chunk_size <= store.CHUNK:
        raise ValueError("chunk_size must be between 1 and 1024")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    sources = _prefixes(source_directory, prefixes, chromosomes)
    annotation_sources = _annotation_sources(annotation_directory, sources)
    output = Path(output_directory).resolve()
    if output.exists() and not resume:
        raise FileExistsError("prepared output already exists; use resume for its identical source")
    if (output / "dataset.json").exists():
        raise FileExistsError("completed prepared datasets are immutable")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.perf_counter()
    contexts = []
    population = None
    for chromosome, prefix in sources.items():
        descriptor, pairs = _source_descriptor(chromosome, prefix,
            annotation_sources[chromosome], output, sample_pairs, variant_indices, chunk_size)
        if population is None:
            population = pairs
        elif not np.array_equal(population, pairs):
            raise ValueError("selected FID/IID order differs between source chromosomes")
        contexts.append(descriptor)
    schedule = _preparation_schedule(contexts, cpu_threads, memory_limit_gib, max_frames=max_frames)
    preflight_started = time.perf_counter()
    if schedule["preflight_workers"] > 1:
        contexts = _process_groups(_preflight_group, contexts, schedule["preflight_workers"],
            schedule["preflight_cpu_threads_per_worker"])
    else:
        contexts = [_preflight_chromosome(context) for context in contexts]
    schedule["preflight_seconds"] = time.perf_counter() - preflight_started
    analysis = contexts[0]["analysis"]
    # No Writer exists yet. Check every chromosome again after independent
    # annotation jobs have finished, before permitting any genotype decoding.
    for context in contexts:
        _check_sources(context)
        source_pairs = _read_fam(_plink_file(context["prefix"], ".fam"))
        selected_pairs, selected_rows = _selected_samples(source_pairs, sample_pairs)
        if (not np.array_equal(population, selected_pairs) or
                not np.array_equal(np.load(context["pairs_file"], mmap_mode="r", allow_pickle=False), population) or
                not np.array_equal(np.load(context["sample_rows_file"], mmap_mode="r", allow_pickle=False), selected_rows)):
            raise ValueError("selected FID/IID order changed during chromosome preflight")
        if context["analysis"] != analysis:
            raise ValueError("chromosome annotation catalogs differ")
    preparation = dict(schema_version=1, source_format="PLINK-BED-BIM-FAM",
        chromosomes={context["chromosome"]: context["binding"] for context in contexts},
        chunk_size=chunk_size, hardlink_annotations=bool(hardlink_annotations))
    pending = output / "preparation.pending.json"
    if pending.exists():
        if json.loads(pending.read_text()) != preparation:
            raise ValueError("resume preparation source/sample/annotation binding differs")
        try:
            saved_pairs = np.load(output / "sample_pairs.npy", allow_pickle=False)
            saved_keys = np.load(output / "sample_ids.npy", allow_pickle=False)
        except (OSError, ValueError):
            raise ValueError("resume root sample axis is missing or invalid") from None
        if (not np.array_equal(saved_pairs, population) or
                not np.array_equal(saved_keys, sample_keys(population))):
            raise ValueError("resume root sample pairs or identity keys differ from the selected population")
    else:
        store.atomic(pending, _canonical(preparation))
        np.save(output / "sample_pairs.npy", population, allow_pickle=False)
        np.save(output / "sample_ids.npy", sample_keys(population), allow_pickle=False)
    options = dict(cpu_threads=schedule["conversion_cpu_threads_per_worker"],
        hardlink_annotations=bool(hardlink_annotations), chunk_size=chunk_size)
    conversion_started = time.perf_counter()
    written_frames = 0
    if schedule["conversion_workers"] > 1:
        converted = _process_groups(_conversion_group, contexts, schedule["conversion_workers"],
            schedule["conversion_cpu_threads_per_worker"], options=options)
        written_frames = sum(result["committed_frames"] for result in converted)
    else:
        converted = []
        for context in contexts:
            remaining = None if max_frames is None else max_frames - written_frames
            result = _convert_chromosome(context, max_frames=remaining, **options)
            written_frames += result["committed_frames"]
            if not result["completed"]:
                schedule["conversion_seconds"] = time.perf_counter() - conversion_started
                storage = _preparation_storage(output, contexts, completed=False)
                return dict(completed=False, dataset=None, committed_frames=written_frames,
                    seconds=time.perf_counter()-started, checkpoint_directory=str(output),
                    cpu_threads=cpu_threads, parallel_execution=schedule, storage=storage)
            converted.append(result)
    schedule["conversion_seconds"] = time.perf_counter() - conversion_started
    for context in contexts:
        _check_sources(context)
    entries = [result["entry"] for result in converted]
    metrics = [result["metrics"] for result in converted]
    dataset = dict(schema_version=2, sample_identifier_format="fid_iid_json",
        sample_pairs="sample_pairs.npy", sample_ids="sample_ids.npy", chromosomes=entries,
        annotation_catalog=analysis["annotation_catalog"], annotation_names=analysis["annotation_names"],
        qc_path=analysis["qc_path"], preparation=dict(source_format="PLINK-BED-BIM-FAM",
            completed=True, variant_subset=any(context["variant_subset"] for context in contexts),
            cpu_threads=cpu_threads, memory_limit_gib=memory_limit_gib, chunk_size=chunk_size,
            seconds=time.perf_counter()-started, chromosomes=metrics, parallel_execution=schedule,
            storage_report="storage.report.json"))
    raw = _canonical(dataset)
    # The caller process alone publishes the cross-chromosome commit. Each
    # child has already committed and validated its independent immutable files.
    store.atomic(output / "dataset.json", raw)
    store.atomic(output / "COMPLETE", hashlib.sha256(raw).hexdigest().encode())
    before_cleanup = directory_storage(output, exclude=("storage.report.json",))
    cleanup = _cleanup_committed_staging(output, raw)
    storage = _preparation_storage(output, contexts, completed=True,
        before_cleanup=before_cleanup, cleanup=cleanup)
    return dict(completed=True, dataset=str(output / "dataset.json"), n_samples=len(population),
        chromosomes=metrics, seconds=time.perf_counter()-started,
        committed_frames=written_frames, cpu_threads=cpu_threads, parallel_execution=schedule,
        storage=storage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--annotation-directory", type=Path, required=True)
    parser.add_argument("--chromosomes", nargs="+")
    parser.add_argument("--sample-pairs", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--memory-limit-gib", type=float, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copy-annotations", action="store_true")
    args = vars(parser.parse_args())
    args["hardlink_annotations"] = not args.pop("copy_annotations")
    print(json.dumps(prepare_WGS_data(**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
