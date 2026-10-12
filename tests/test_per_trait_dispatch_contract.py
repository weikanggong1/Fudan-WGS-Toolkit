"""Canonical CSV owners stay independent of GPU count; legacy plans persist."""
import pytest

from fudan_wgs_toolkit.phewas_run import _association_partition_strategy


@pytest.mark.parametrize('plan', [
    {'public_api_options': True},
    {'csv_output_layout': 'per_trait_checkpoint'},
    {'csv_output_layout': 'per_trait_checkpoint', 'association_partition_strategy': 'auto'},
    {'csv_output_layout': 'per_trait_checkpoint', 'association_partition_strategy': 'cohort_tiles'},
])
def test_canonical_output_has_one_active_owner_per_fixed_trait_tile(plan):
    assert _association_partition_strategy(plan) == 'cohort_tiles'


@pytest.mark.parametrize('plan', [
    {'public_api_options': True, 'association_partition_strategy': 'whole_chromosomes'},
    {'csv_output_layout': 'per_trait_checkpoint',
     'association_partition_strategy': 'whole_chromosomes'},
])
def test_whole_chromosomes_preserve_disjoint_canonical_owners(plan):
    assert _association_partition_strategy(plan) == 'whole_chromosomes'


@pytest.mark.parametrize('strategy', ['genetic_jobs', 'ordered_cohorts', 'unknown'])
def test_incompatible_explicit_canonical_dispatch_fails_closed(strategy):
    with pytest.raises(ValueError, match='requires cohort_tiles or whole_chromosomes dispatch'):
        _association_partition_strategy({'public_api_options': True,
                                        'association_partition_strategy': strategy})


@pytest.mark.parametrize('strategy', ['auto', 'genetic_jobs', 'ordered_cohorts', 'cohort_tiles'])
def test_existing_mixed_shard_private_plans_preserve_dispatch(strategy):
    assert _association_partition_strategy({'association_partition_strategy': strategy}) == strategy
