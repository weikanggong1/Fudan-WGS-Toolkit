"""Private aggregate calibration profiles and an explicitly artificial preset."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from .masks import CODING_CATEGORIES_INCL_PTV, NONCODING_CATEGORIES

PROFILE_FORMAT = "synthetic-wgs-calibration-v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def normalize_profile(profile):
    """Validate statistics without accepting individual identifiers or loci."""
    if not isinstance(profile, dict) or profile.get("format") != PROFILE_FORMAT:
        raise ValueError("calibration profile must use synthetic-wgs-calibration-v1")
    reference_n = _integer(profile.get("reference_n"), "reference_n", 128)
    entries = profile.get("chromosomes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("calibration profile needs chromosome entries")
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("calibration chromosome must be a dictionary")
        chromosome = str(entry.get("name", "")).removeprefix("chr")
        if chromosome not in {str(value) for value in range(1, 23)} or chromosome in seen:
            raise ValueError("calibration chromosomes must be distinct autosomes")
        seen.add(chromosome)
        variants = _integer(entry.get("variants"), "variants", 1)
        bins = entry.get("frequency_bins")
        if not isinstance(bins, list) or not bins:
            raise ValueError("each chromosome needs a frequency spectrum")
        frequencies = []
        for item in bins:
            if not isinstance(item, dict):
                raise ValueError("frequency bins must be dictionaries")
            count = _integer(item.get("count"), "frequency bin count")
            maf, missing = item.get("maf"), item.get("missing_rate")
            if (isinstance(maf, bool) or not isinstance(maf, (int, float)) or not math.isfinite(maf)
                    or not 0 <= maf <= 0.5):
                raise ValueError("frequency-bin maf must be finite in [0, .5]")
            if (isinstance(missing, bool) or not isinstance(missing, (int, float)) or not math.isfinite(missing)
                    or not 0 <= missing <= 1):
                raise ValueError("frequency-bin missing_rate must be finite in [0, 1]")
            normalized = dict(count=count, maf=float(maf), missing_rate=float(missing))
            for key in ("minor_ac_mean", "called_alleles_mean", "minor_ac_low", "minor_ac_high"):
                if key in item:
                    value = item[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("optional allele-count summaries must be nonnegative and finite")
                    normalized[key] = float(value)
            called_mean = normalized.get("called_alleles_mean", 2*reference_n*(1-float(missing)))
            if called_mean > 2*reference_n:
                raise ValueError("called_alleles_mean exceeds the reference diploid sample axis")
            if normalized.get("minor_ac_mean", 0) > called_mean/2:
                raise ValueError("minor_ac_mean exceeds half the reference called alleles")
            if any(normalized.get(key, 0) > reference_n for key in ("minor_ac_low", "minor_ac_high")):
                raise ValueError("minor allele count bounds exceed the reference sample count")
            if ("minor_ac_low" in normalized and "minor_ac_high" in normalized and
                    normalized["minor_ac_low"] > normalized["minor_ac_high"]):
                raise ValueError("minor allele count lower bound exceeds its upper bound")
            frequencies.append(normalized)
        if sum(item["count"] for item in frequencies) != variants:
            raise ValueError("frequency-bin counts must equal physical variant count")
        masks, keys = [], set()
        if not isinstance(entry.get("masks"), list):
            raise ValueError("each chromosome requires an explicit mask list")
        for item in entry["masks"]:
            if not isinstance(item, dict):
                raise ValueError("calibration masks must be dictionaries")
            kind, category = item.get("kind"), item.get("category")
            allowed = CODING_CATEGORIES_INCL_PTV if kind == "coding" else NONCODING_CATEGORIES if kind == "noncoding" else ("ncRNA",) if kind == "ncrna" else ()
            if category not in allowed:
                raise ValueError("calibration mask kind/category is not canonical")
            group = _integer(item.get("group"), "anonymous mask group")
            m = _integer(item.get("m"), "published mask m", 2)
            key = kind, group, category
            if key in keys:
                raise ValueError("calibration mask entries must be unique")
            keys.add(key)
            masks.append(dict(kind=kind, group=group, category=category, m=m))
        chromosome_profile = dict(name=chromosome, variants=variants, frequency_bins=frequencies, masks=masks)
        if "position_span" in entry:
            chromosome_profile["position_span"] = _integer(entry["position_span"], "position_span", variants)
        result.append(chromosome_profile)
    normalized = dict(format=PROFILE_FORMAT, reference_n=reference_n,
                      chromosomes=sorted(result, key=lambda value: int(value["name"])))
    if "analysis_reference_n" in profile:
        normalized["analysis_reference_n"] = _integer(profile["analysis_reference_n"], "analysis_reference_n", 1)
    return normalized


def small_profile():
    """Return artificial scale/frequency/mask counts; never a cohort default."""
    masks = [dict(kind="coding", group=0, category=category, m=m) for category, m in
        (("plof", 3), ("plof_ds", 5), ("missense", 4), ("disruptive_missense", 2),
         ("synonymous", 3), ("ptv", 5), ("ptv_ds", 7))]
    masks += [dict(kind="noncoding", group=1, category=category, m=4)
              for category in NONCODING_CATEGORIES]
    masks.append(dict(kind="ncrna", group=2, category="ncRNA", m=5))
    return normalize_profile(dict(format=PROFILE_FORMAT, reference_n=1000,
        chromosomes=[dict(name=str(chromosome), variants=1024,
            frequency_bins=[dict(count=600, maf=0.0005, missing_rate=0.0),
                dict(count=300, maf=0.003, missing_rate=0.0),
                dict(count=100, maf=0.02, missing_rate=0.0),
                dict(count=24, maf=0.2, missing_rate=0.0)], masks=masks)
                     for chromosome in range(1, 23)]))


def resolve_profile(calibration_profile=None, preset="reference"):
    """Resolve explicit input, environment or private user configuration.

    User configuration is JSON at FUDAN_WGS_SIMULATION_CONFIG or
    ~/.config/fudan-wgs-toolkit/simulation.json, with calibration_profile as
    its profile filename. Relative filenames resolve beside that config.
    A missing reference profile never falls back to an invented full scale.
    """
    if preset not in ("reference", "small"):
        raise ValueError("preset must be reference or small")
    if preset == "small":
        if calibration_profile is not None:
            raise ValueError("small preset cannot also select a calibration profile")
        profile = small_profile()
        raw = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
        return profile, dict(kind="artificial_small_preset", calibrated=False,
                             profile_sha256=hashlib.sha256(raw).hexdigest())
    selected, source = calibration_profile, "explicit"
    if selected is None:
        selected, source = os.environ.get("FUDAN_WGS_SIMULATION_PROFILE"), "environment"
    if selected is None:
        configuration = Path(os.environ.get("FUDAN_WGS_SIMULATION_CONFIG", "~/.config/fudan-wgs-toolkit/simulation.json")).expanduser()
        if configuration.is_file():
            document = json.loads(configuration.read_text(encoding="utf-8"))
            selected = document.get("calibration_profile")
            if selected is not None:
                selected = Path(selected).expanduser()
                if not selected.is_absolute():
                    selected = configuration.parent / selected
                source = "user_configuration"
    if selected is None:
        raise FileNotFoundError("reference calibration profile is not configured; supply calibration_profile, set FUDAN_WGS_SIMULATION_PROFILE, or explicitly choose preset='small'")
    if isinstance(selected, dict):
        profile = normalize_profile(selected)
        raw = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    else:
        path = Path(selected).expanduser().resolve()
        raw = path.read_bytes()
        profile = normalize_profile(json.loads(raw))
    return profile, dict(kind="private_aggregate_calibration", calibrated=True,
        resolved_from=source, profile_sha256=hashlib.sha256(raw).hexdigest())
