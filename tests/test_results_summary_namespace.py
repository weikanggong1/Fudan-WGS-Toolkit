"""Disruptive-missense summaries preserve the incoming result schema."""
import math

import pytest

from fudan_wgs_toolkit.results import add_disruptive_missense, assemble_phewas_results


ELEMENTARY = ("SKAT(1,25)", "SKAT(1,1)", "Burden(1,25)", "Burden(1,1)",
              "ACAT-V(1,25)", "ACAT-V(1,1)")
SUMMARIES = ("S(1,25)", "S(1,1)", "B(1,25)", "B(1,1)", "A(1,25)", "A(1,1)")


def rows(namespace="WGS", *, binary=True):
    selected = (2, 3) if binary else range(6)
    missense = {"Gene name": "gene", "Chr": "21", "Category": "missense", "#SNV": 10}
    disruptive = {"Gene name": "gene", "Chr": "21", "Category": "disruptive_missense", "#SNV": 2}
    for index in selected:
        base = ELEMENTARY[index]
        missense[base] = .1 + index * .01
        missense[base + "-annotation"] = .2 + index * .01
        missense[namespace + "-" + SUMMARIES[index]] = .9
        disruptive[base] = .03 + index * .01
    missense[namespace + ("-B" if binary else "-O")] = .8
    if not binary:
        missense["ACAT-O"] = .77
    return missense, disruptive


def combiner(values):
    # A transparent reduction exposes exactly which inputs reach the CCT.
    return sum(values) / len(values)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("namespace", ["WGS", "STAAR"])
def test_combined_summary_updates_existing_keys_without_stale_alias(binary, namespace):
    missense, disruptive = rows(namespace, binary=binary)
    original = dict(missense)
    combined = add_disruptive_missense(missense, disruptive, use_spa=binary,
        cauchy_combiner=combiner)
    other = "STAAR" if namespace == "WGS" else "WGS"
    assert not any(key.startswith(other + "-") for key in combined)
    assert missense == original
    selected = (2, 3) if binary else range(6)
    inputs = []
    for index in selected:
        base = ELEMENTARY[index]
        values = [missense[base], missense[base + "-annotation"], disruptive[base]]
        assert combined[namespace + "-" + SUMMARIES[index]] == pytest.approx(combiner(values))
        assert combined[base + "-Disruptive"] == disruptive[base]
        inputs.extend(values)
    assert combined[namespace + ("-B" if binary else "-O")] == pytest.approx(combiner(inputs))
    if not binary:
        assert combined["ACAT-O"] == .77


@pytest.mark.parametrize("namespace", ["WGS", "STAAR"])
def test_no_disruptive_result_preserves_summaries_and_adds_ones(namespace):
    missense, _ = rows(namespace)
    combined = add_disruptive_missense(missense, None, use_spa=True)
    for key, value in missense.items():
        assert combined[key] == value
    assert combined["Burden(1,25)-Disruptive"] == 1
    assert combined["Burden(1,1)-Disruptive"] == 1


@pytest.mark.parametrize("namespace", ["WGS", "STAAR"])
def test_binary_nan_one_and_zero_filtering_retains_author_convention(namespace):
    missense, disruptive = rows(namespace)
    for base in ELEMENTARY[2:4]:
        missense[base] = float("nan")
        missense[base + "-annotation"] = 1
        disruptive[base] = float("nan")
    no_positive = add_disruptive_missense(missense, disruptive, use_spa=True,
        cauchy_combiner=lambda _: pytest.fail("empty/all-zero CCT must not execute"))
    assert no_positive[namespace + "-B"] == 1
    assert no_positive[namespace + "-B(1,25)"] == 1
    assert no_positive["Burden(1,25)-Disruptive"] == 1
    missense["Burden(1,25)"] = 0
    all_zero = add_disruptive_missense(missense, disruptive, use_spa=True,
        cauchy_combiner=lambda _: pytest.fail("empty/all-zero CCT must not execute"))
    assert all_zero[namespace + "-B"] == 1
    disruptive["Burden(1,1)"] = .025
    combined = add_disruptive_missense(missense, disruptive, use_spa=True,
        cauchy_combiner=combiner)
    assert combined[namespace + "-B"] == pytest.approx(.0125)


@pytest.mark.parametrize("key", ["WGS-B(1,25)", "WGS-B(1,1)", "WGS-B"])
def test_conflicting_aliases_fail_before_emitting_a_combined_row(key):
    missense, disruptive = rows()
    missense["STAAR-" + key[4:]] = .1234
    with pytest.raises(ValueError, match="conflicting"):
        add_disruptive_missense(missense, disruptive, use_spa=True)
    assert missense[key] in (.9, .8)


@pytest.mark.parametrize("value", [.9, float("nan")])
def test_identical_aliases_are_collapsed_before_enhancement(value):
    missense, disruptive = rows()
    missense["WGS-B(1,25)"] = missense["STAAR-B(1,25)"] = value
    combined = add_disruptive_missense(missense, disruptive, use_spa=True,
        cauchy_combiner=combiner)
    assert "STAAR-B(1,25)" not in combined
    assert math.isfinite(combined["WGS-B(1,25)"])


def test_private_runtime_assembly_keeps_a_single_enhanced_namespace():
    missense, disruptive = rows("STAAR")
    assembled = assemble_phewas_results([[missense, disruptive]], kind="coding",
        use_spa=True, cauchy_combiner=combiner)
    result = assembled["missense"][0][0]
    assert not any(key.startswith("WGS-") for key in result)
    assert result["STAAR-B(1,25)"] != missense["STAAR-B(1,25)"]
    assert result["STAAR-B"] != missense["STAAR-B"]
