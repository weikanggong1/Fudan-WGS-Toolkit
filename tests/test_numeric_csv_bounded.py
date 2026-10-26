"""Static CSV contracts and allocation admission, not scientific benchmarks."""
import csv
import importlib

import numpy as np
import pytest
from _file_identity_fixture import rewrite_with_changed_ctime

run = importlib.import_module("fudan_wgs_toolkit.run")
resources = importlib.import_module("fudan_wgs_toolkit.phewas_resources")
GiB = resources.GiB


def constraint(limit=200 * GiB, current=0, *, inactive=0, scope="cgroup_v2"):
    return dict(limit_bytes=limit, current_bytes=current,
                inactive_file_bytes=inactive, effective_current_bytes=max(0, current - inactive),
                scope=scope)


@pytest.fixture(autouse=True)
def memory_available(monkeypatch):
    observed = constraint()
    monkeypatch.setattr(resources, "read_host_memory", lambda: dict(observed, constraints=[observed]))


def inputs(tmp_path, phenotype="FID,IID,trait_a,trait_b\n0,001,1,NA\nf,001,0,1\n"):
    p, c = tmp_path / "phenotypes.csv", tmp_path / "covariates.csv"
    p.write_text(phenotype, encoding="utf-8")
    c.write_text("FID,IID,age,sex\n0,001,30,0\nf,001,40,1\n", encoding="utf-8")
    return p, c


def test_float64_missing_ids_order_and_narrow_metadata(tmp_path):
    p, c = inputs(tmp_path, "\ufeff trait_a ,IID,FID,trait_b\n1,001,0,NA\n\n0,001,f, 1e-300 \n ,, , \n")
    c.write_text("IID,FID\n001,f\n001,0\n", encoding="utf-8")
    phenotype, covariate = run.read_csv_inputs(p, c)
    assert phenotype.names == ("trait_a", "trait_b")
    assert phenotype.values.dtype == np.float64
    np.testing.assert_array_equal(phenotype.sample_pairs, [["0", "001"], ["f", "001"]])
    np.testing.assert_array_equal(phenotype.values[:, 0], [1, 0])
    assert np.isnan(phenotype.values[0, 1]) and phenotype.values[1, 1] == 1e-300
    assert covariate.names == () and covariate.values.shape == (2, 0)
    layout = run._csv_layout(p, require_columns=True)
    assert layout["rows"] == 2 and layout["identifier_width"] == 3
    assert layout["array_bytes"] == phenotype.values.nbytes + phenotype.sample_pairs.nbytes
    assert layout["identity"] == run._csv_stat(p)


def test_unicode_long_identifiers_are_not_truncated(tmp_path):
    fid = "家庭" * 80
    p, c = inputs(tmp_path, f"FID,IID,trait_a,trait_b\n{fid},00000000000001,0,1\n0,0002,1,0\n")
    parsed = run._read_csv(p, require_columns=True)
    assert parsed.sample_pairs[0, 0] == fid
    assert parsed.sample_pairs[0, 1] == "00000000000001"


@pytest.mark.parametrize("token", ["", "na", "NaN", "n/a", "none", "NULL"])
def test_missing_tokens_remain_nan(tmp_path, token):
    p, c = inputs(tmp_path, f"FID,IID,trait_a,trait_b\n0,001,{token},0\n")
    assert np.isnan(run.read_csv_inputs(p, c)[0].values[0, 0])


@pytest.mark.parametrize("token", ["inf", "-inf", "Infinity", "+nan", "1e999"])
def test_infinite_or_nonstandard_nonfinite_numeric_tokens_rejected(tmp_path, token):
    p, c = inputs(tmp_path, f"FID,IID,trait_a,trait_b\n0,001,{token},0\n")
    with pytest.raises(ValueError, match="infinity"):
        run.read_csv_inputs(p, c)


@pytest.mark.parametrize("contents,match", [
    ("", "missing CSV header"),
    ("FID,IID,trait_a,trait_a\n0,1,0,1\n", "nonempty and unique"),
    ("FID,IID,\n0,1,0\n", "nonempty and unique"),
    ("IID,trait\n1,0\n", "FID and IID"),
    ("FID,IID\n0,1\n", "at least one named phenotype"),
    ("FID,IID,trait\n\n", "no participant rows"),
    ("FID,IID,trait\n0,1,1,2\n", "number of columns"),
    ("FID,IID,trait\n0,1,abc\n", "numeric or missing"),
    ("FID,IID,trait\n ,1,0\n", "without whitespace"),
    ("FID,IID,trait\n0,a b,0\n", "without whitespace"),
    ("FID,IID,trait\n0,1,0\nf,1,1\n0,1,1\n", "duplicate FID/IID"),
])
def test_invalid_csv_contracts(tmp_path, contents, match):
    p, c = inputs(tmp_path, contents)
    with pytest.raises(ValueError, match=match):
        run.read_csv_inputs(p, c)


def test_file_mutation_between_passes_rejected_even_with_same_size_and_mtime(tmp_path, monkeypatch):
    p, c = inputs(tmp_path)
    scan = run._csv_layout

    def mutate(path, **kwargs):
        result = scan(path, **kwargs)
        if path == p:
            rewrite_with_changed_ctime(p, p.read_bytes().replace(b"f,001,0,1", b"f,001,1,1"),
                                      preserve_mtime=True)
        return result

    monkeypatch.setattr(run, "_csv_layout", mutate)
    with pytest.raises(ValueError, match="CSV changed"):
        run.read_csv_inputs(p, c)


def test_earlier_input_mutation_while_reading_later_input_rejected(tmp_path, monkeypatch):
    p, c = inputs(tmp_path)
    read = run._read_csv

    def mutate(path, **kwargs):
        result = read(path, **kwargs)
        if path == c:
            p.write_text(p.read_text() + "0,002,1,0\n")
        return result

    monkeypatch.setattr(run, "_read_csv", mutate)
    with pytest.raises(ValueError, match="CSV changed"):
        run.read_csv_inputs(p, c)


def test_both_input_arrays_admitted_before_any_allocation(tmp_path, monkeypatch):
    p, c = inputs(tmp_path)
    layouts = [run._csv_layout(p, require_columns=True), run._csv_layout(c, require_columns=False)]
    required = sum(item["array_bytes"] for item in layouts) + max(item["workspace_bytes"] for item in layouts)
    seen = constraint(current=200 * GiB - 20 * GiB - required + 1)
    monkeypatch.setattr(resources, "read_host_memory", lambda: dict(seen, constraints=[seen]))
    monkeypatch.setattr(run.np, "empty", lambda *args, **kwargs: pytest.fail("allocation preceded admission"))
    with pytest.raises(resources.HostMemoryDeferred) as error:
        run.read_csv_inputs(p, c)
    assert error.value.required_bytes == required


def test_inactive_file_reclamation_and_default_reserve_use_mature_policy(monkeypatch):
    seen = constraint(current=199 * GiB, inactive=45 * GiB)
    monkeypatch.setattr(resources, "read_host_memory", lambda: dict(seen, constraints=[seen]))
    report = run._csv_host_admission(1 * GiB)
    assert report["available_bytes"] == 26 * GiB
    assert report["reserve_bytes"] == 20 * GiB
    seen["inactive_file_bytes"] = 0
    seen["effective_current_bytes"] = seen["current_bytes"]
    with pytest.raises(resources.HostMemoryDeferred):
        run._csv_host_admission(1 * GiB)


def test_wider_ancestors_and_physical_available_memory_are_checked(monkeypatch):
    checks = [constraint(current=150 * GiB), constraint(250 * GiB, 229 * GiB),
              constraint(500 * GiB, 490 * GiB, scope="system_available_memory")]
    monkeypatch.setattr(resources, "read_host_memory", lambda: dict(checks[0], constraints=checks))
    with pytest.raises(resources.HostMemoryDeferred):
        run._csv_host_admission(1 * GiB)
    checks[-1]["effective_current_bytes"] = 450 * GiB
    assert run._csv_host_admission(1 * GiB)["available_bytes"] == GiB
    with pytest.raises(resources.HostMemoryDeferred):
        run._csv_host_admission(2 * GiB)


def test_requested_limit_does_not_apply_to_wider_ancestor_usage(monkeypatch):
    checks = [constraint(current=5 * GiB), constraint(500 * GiB, 300 * GiB)]
    monkeypatch.setattr(resources, "read_host_memory", lambda: dict(checks[0], constraints=checks))
    report = run._csv_host_admission(GiB, host_memory_limit_gib=12, host_memory_reserve_gib=3)
    assert report["resolved_limit_bytes"] == 12 * GiB and report["available_bytes"] == 4 * GiB


@pytest.mark.parametrize("options", [dict(host_memory_limit_gib=True), dict(host_memory_limit_gib=0),
    dict(host_memory_limit_gib=float("inf")), dict(host_memory_reserve_gib=-1),
    dict(host_memory_limit_gib=1, host_memory_reserve_gib=1)])
def test_invalid_host_options_fail_before_allocations(options):
    with pytest.raises(ValueError):
        run._csv_host_admission(1, **options)


def test_capacity_failure_precedes_allocation_and_tiny_nodes_keep_proportional_reserve(monkeypatch):
    seen = constraint(limit=2 * GiB)
    monkeypatch.setattr(resources, "read_host_memory", lambda: dict(seen, constraints=[seen]))
    report = run._csv_host_admission(1)
    assert report["reserve_bytes"] == (2 * GiB) // 10
    with pytest.raises(resources.HostMemoryRequestTooLarge):
        run._csv_host_admission(2 * GiB)


def test_count_pass_does_not_parse_or_retain_all_numeric_values(tmp_path, monkeypatch):
    p, c = inputs(tmp_path)
    with p.open("a", newline="") as stream:
        csv.writer(stream).writerow(["0", "002", "bad", 1])
    layout = run._csv_layout(p, require_columns=True)
    assert layout["rows"] == 3
    assert not any(isinstance(value, np.ndarray) for value in layout.values())
    with pytest.raises(ValueError, match="numeric or missing"):
        run.read_csv_inputs(p, c)
