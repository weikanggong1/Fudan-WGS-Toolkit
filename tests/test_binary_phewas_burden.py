"""Small-burden covariance and selective SPA preserve the reference formula."""
import numpy as np
import pytest
import torch
from contextlib import contextmanager
from fudan_wgs_toolkit.binary import (staar_binary_phewas, association_binary_spa,
    prepare_binary_burdens, binary_single_phewas, individual_score_test_spa)
from fudan_wgs_toolkit.phewas_models import prepare_phewas_model
from fudan_wgs_toolkit.statistics import annotation_weights,_chi1_from_score
from fudan_wgs_toolkit.binary_null import (compact_spa_state_from_fitted_arrays,
    compact_binary_normal_from_fitted_arrays, ImmutableFittedBinarySource)
from fudan_wgs_toolkit.phewas_models import SparseKinshipData


def fixture():
    n=64;x=np.column_stack((np.ones(n),np.linspace(-1,1,n)))
    y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],8)
    prepared=prepare_phewas_model(y,np.arange(n).astype(str),x,association_mode="fp64")
    g=torch.zeros((n,9),dtype=torch.float64)
    for j in range(9):g[(j*5+np.arange(6))%n,j]=1
    maf=torch.linspace(.001,.009,9,dtype=torch.float64)
    annotation=torch.arange(18,dtype=torch.float64).reshape(9,2)/3
    return prepared,g,maf,annotation


def test_binary_projects_burdens_only_and_matches_full_covariance(monkeypatch):
    prepared,g,f,a=fixture();model=prepared.model
    u,v=model.score_covariance(g)
    weights=annotation_weights(f,a)[0]
    expected=_chi1_from_score(weights.T@u,(weights.T@v@weights).diagonal())
    original=model.score_covariance;shapes=[]
    def record(block):
        shapes.append(tuple(block.shape));return original(block)
    monkeypatch.setattr(model,"score_covariance",record)
    result,diag=staar_binary_phewas(model,g,f,a,["a","b"],return_diagnostics=True)
    assert shapes == [(64,6)] and diag["covariance_shape"] == [6,6]
    assert not diag["variant_covariance_constructed"] and diag["normal_only"]
    torch.testing.assert_close(diag["pvalues"],expected,atol=1e-12,rtol=1e-12)
    assert set(result)=={"num_variant","cMAC","Burden(1,25)","Burden(1,1)",
        "Burden(1,25)-a","Burden(1,25)-b","Burden(1,1)-a","Burden(1,1)-b",
        "STAAR-B(1,25)","STAAR-B(1,1)","STAAR-B"}


@pytest.mark.parametrize('layout', ['c', 'f'])
def test_bounded_validation_preserves_complete_binary_probabilities_and_source(layout, monkeypatch):
    """Run the actual burden/projection/selective SPA path with bounded checks."""
    from fudan_wgs_toolkit import statistics, tensor_validation
    prepared, g, f, a = fixture()
    if layout == 'f':
        g = g.T.contiguous().T
    before = g.clone()
    stamp = (g.stride(), g._version, g.untyped_storage().data_ptr())
    expected, expected_diagnostics = staar_binary_phewas(prepared.model, g, f, a,
        ['a', 'b'], spa_model=prepared.spa_model, p_filter_cutoff=1, return_diagnostics=True)
    checked = []
    def bounded(value):
        checked.append(tuple(value.shape))
        return tensor_validation._all_finite_blocked(value, 13)
    monkeypatch.setattr(statistics, 'tensor_all_finite', bounded)
    actual, diagnostic = staar_binary_phewas(prepared.model, g, f, a,
        ['a', 'b'], spa_model=prepared.spa_model, p_filter_cutoff=1, return_diagnostics=True)
    assert tuple(g.shape) in checked
    assert actual == expected
    for field in ('pvalues', 'nominal_pvalues', 'spa_selected'):
        torch.testing.assert_close(diagnostic[field], expected_diagnostics[field], rtol=0, atol=0)
    torch.testing.assert_close(g, before, rtol=0, atol=0)
    assert (g.stride(), g._version, g.untyped_storage().data_ptr()) == stamp


def test_selective_spa_matches_mature_binary_wrapper_when_all_selected():
    prepared,g,f,a=fixture();spa=prepared.spa_model
    expected=association_binary_spa(g,f,spa.scaled_residuals,spa.fitted_probability,
        spa.xw,spa.projection_left,a,["a","b"])
    # Public main uses WGS keys; private PheWAS retains canonical STAAR keys.
    expected={key.replace("WGS-B", "STAAR-B"):value for key,value in expected.items()}
    result,diag=staar_binary_phewas(prepared.model,g,f,a,["a","b"],spa_model=spa,
        p_filter_cutoff=1,return_diagnostics=True)
    torch.testing.assert_close(diag["spa_selected"],diag["nominal_pvalues"]<1,atol=0,rtol=0)
    assert diag["spa_selected"].any() and not diag["normal_only"]
    # Burden-first association reorders mathematically equivalent FP64
    # products; SPA's root iteration can amplify those final ulps slightly.
    for key in expected:assert result[key] == pytest.approx(expected[key],abs=1e-10,rel=1e-10)


def test_spa_filter_preserves_unselected_nominal_pvalues():
    prepared,g,f,a=fixture()
    result,diag=staar_binary_phewas(prepared.model,g,f,a,["a","b"],
        spa_model=prepared.spa_model,p_filter_cutoff=1e-12,return_diagnostics=True)
    assert not diag["spa_selected"].any()
    torch.testing.assert_close(diag["pvalues"],diag["nominal_pvalues"],atol=0,rtol=0)


def test_spa_rejects_a_different_phenotype_on_same_ids():
    prepared,g,f,a=fixture()
    prepared.spa_model.phenotype=1-prepared.spa_model.phenotype
    with pytest.raises(ValueError,match="phenotype differs"):
        staar_binary_phewas(prepared.model,g,f,a,spa_model=prepared.spa_model)


def test_host_tiled_burdens_reuse_matches_direct_small_product(monkeypatch):
    prepared,g,f,a=fixture();model=prepared.model
    cached=prepare_binary_burdens(model,g,f,a,["a","b"],variant_tile_size=2)
    weights=annotation_weights(f,a)[0]
    torch.testing.assert_close(cached.burdens,g@weights,atol=1e-12,rtol=1e-12)
    def forbidden(*args,**kwargs):raise AssertionError("recomputed prepared burdens")
    monkeypatch.setattr("fudan_wgs_toolkit.binary.prepare_binary_burdens",forbidden)
    result=staar_binary_phewas(model,g,f,a,["a","b"],prepared_burdens=cached)
    assert result["num_variant"]==9
    with pytest.raises(ValueError,match="selected MAF"):
        staar_binary_phewas(model,g,f+.00001,a,["a","b"],prepared_burdens=cached)
    g[0,0]=1
    with pytest.raises(ValueError,match="different source"):
        staar_binary_phewas(model,g,f,a,["a","b"],prepared_burdens=cached)


def test_single_selected_spa_matches_mature_path_and_keeps_nominal_columns():
    prepared,g,f,a=fixture();model=prepared.model;spa=prepared.spa_model
    scores,variance=model.individual_score_variance(g)
    nominal=_chi1_from_score(scores,variance)
    expected=individual_score_test_spa(g,spa.scaled_residuals,spa.fitted_probability,
        spa.xw,spa.projection_left,normal_pvalues=nominal,p_filter_cutoff=.99)
    result=binary_single_phewas(model,g,spa_model=spa,normal_score=scores,
        normal_variance=variance,normal_pvalues=nominal,p_filter_cutoff=.99,
        spa_variant_tile_size=2)
    torch.testing.assert_close(result["pvalues"],expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(result["normal_pvalues"],nominal,atol=0,rtol=0)
    assert result["spa_selected"].any()
    assert result["pvalue_log"].shape==scores.shape


def test_thin_single_normal_requires_identical_verified_fit_source():
    prepared,g,f,a=fixture();model=prepared.model
    model.phenotype=None
    result=binary_single_phewas(model,g,spa_model=prepared.spa_model)
    assert not result["normal_only"]
    model.null_fit_source_sha256="1"*64
    with pytest.raises(ValueError,match="fitted-source"):
        binary_single_phewas(model,g,spa_model=prepared.spa_model)


def test_retained_labels_do_not_override_different_fitted_source():
    prepared,g,f,a=fixture()
    assert torch.equal(prepared.model.phenotype,prepared.spa_model.phenotype)
    prepared.model.null_fit_source_sha256="1"*64
    with pytest.raises(ValueError,match="fitted-source"):
        binary_single_phewas(prepared.model,g,spa_model=prepared.spa_model)
    with pytest.raises(ValueError,match="fitted-source"):
        staar_binary_phewas(prepared.model,g,f,a,spa_model=prepared.spa_model)


def test_single_zero_normal_probability_retains_finite_score_based_log_tail():
    prepared,g,f,a=fixture()
    values=binary_single_phewas(prepared.model,g,normal_score=torch.full((9,),40.),
        normal_variance=torch.ones(9))
    assert (values["normal_pvalues"]==0).all()
    assert torch.isfinite(values["normal_pvalue_log10"]).all()
    assert not values["spa_zero_log_unverified"].any()


@pytest.mark.parametrize("mixed", [False,True])
def test_compact_spa_reconstruction_matches_full_fitted_state(mixed):
    prepared,g,f,a=fixture()
    if mixed:
        rows=np.arange(0,128,2,dtype=np.int64)
        kinship=SparseKinshipData(np.arange(128).astype(str),np.ones(128),
            rows,rows+1,np.full(64,.5))
        y=np.tile([0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0],8).astype(float)
        x=np.column_stack((np.ones(128),np.linspace(-1,1,128)))
        prepared=prepare_phewas_model(y,np.arange(128).astype(str),x,kinship=kinship,
            association_mode="fp64")
        g=torch.cat([g,g])
        assert float(prepared.spa_model.theta[1])>0 and not prepared.spa_model.boundary_refit
    original=prepared.spa_model
    compact=compact_spa_state_from_fitted_arrays(sample_ids=original.sample_ids,
        covariates=original.x,residual=original.scaled_residuals,
        fitted_probability=original.fitted_probability,
        fixed_effect_covariance=original.fixed_effect_covariance,
        precision=original.precision,null_fit_source_sha256=original.null_fit_source_sha256,
        verified_fit_source_sha256=prepared.model.null_fit_source_sha256,
        has_kinship=original.has_kinship,iterations=original.iterations)
    assert compact.x.data_ptr()==original.x.data_ptr()  # Shared FP64 design.
    assert compact.xw.data_ptr()==compact.precision_x.data_ptr()
    assert compact.phenotype is None and compact.working_phenotype is None
    torch.testing.assert_close(compact.xw,original.xw,atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(compact.projection_left,original.projection_left,atol=0,rtol=0)
    full=binary_single_phewas(prepared.model,g,spa_model=original,p_filter_cutoff=1)
    regenerated=binary_single_phewas(prepared.model,g,spa_model=compact,p_filter_cutoff=1)
    torch.testing.assert_close(regenerated["pvalues"],full["pvalues"],atol=1e-10,rtol=1e-10)
    full_burden=staar_binary_phewas(prepared.model,g,f,a,spa_model=original,p_filter_cutoff=1)
    compact_burden=staar_binary_phewas(prepared.model,g,f,a,spa_model=compact,p_filter_cutoff=1)
    for key in full_burden:
        assert compact_burden[key]==pytest.approx(full_burden[key],abs=1e-10,rel=1e-10)


def test_compact_spa_refuses_rounded_source_and_different_fit_binding():
    prepared,g,f,a=fixture();original=prepared.spa_model
    args=dict(sample_ids=original.sample_ids,covariates=original.x,
        residual=original.scaled_residuals,fitted_probability=original.fitted_probability,
        fixed_effect_covariance=original.fixed_effect_covariance,precision=original.precision,
        null_fit_source_sha256=original.null_fit_source_sha256,
        verified_fit_source_sha256=prepared.model.null_fit_source_sha256)
    with pytest.raises(ValueError,match="original FP64"):
        compact_spa_state_from_fitted_arrays(**dict(args,covariates=original.x.float()))
    with pytest.raises(ValueError,match="fitted-source"):
        compact_spa_state_from_fitted_arrays(**dict(args,verified_fit_source_sha256="f"*64))
    with pytest.raises(ValueError,match="dense N-by-N"):
        compact_spa_state_from_fitted_arrays(**dict(args,precision=torch.diag(original.precision)))


@pytest.mark.parametrize("kind", ["single","burden"])
@pytest.mark.parametrize("selected", [False,True])
def test_lazy_spa_only_loads_selected_tests_and_preserves_probabilities(kind,selected):
    prepared,g,f,a=fixture();calls=[]
    @contextmanager
    def acquire():
        calls.append("enter")
        try:yield prepared.spa_model
        finally:calls.append("exit")
    cutoff=1 if selected else 1e-12
    if kind=="single":
        expected=binary_single_phewas(prepared.model,g,spa_model=prepared.spa_model,
            p_filter_cutoff=cutoff)
        result=binary_single_phewas(prepared.model,g,spa_acquire=acquire,p_filter_cutoff=cutoff)
    else:
        _,expected=staar_binary_phewas(prepared.model,g,f,a,spa_model=prepared.spa_model,
            p_filter_cutoff=cutoff,return_diagnostics=True)
        _,result=staar_binary_phewas(prepared.model,g,f,a,spa_acquire=acquire,
            p_filter_cutoff=cutoff,return_diagnostics=True)
    assert calls==(["enter","exit"] if selected else [])
    assert not result["normal_only"]
    torch.testing.assert_close(result["pvalues"],expected["pvalues"],atol=0,rtol=0)
    torch.testing.assert_close(result["spa_selected"],expected["spa_selected"],atol=0,rtol=0)


def test_lazy_spa_validates_binding_and_always_closes_factory():
    prepared,g,f,a=fixture();calls=[]
    @contextmanager
    def acquire():
        calls.append("enter")
        try:yield prepared.spa_model
        finally:calls.append("exit")
    prepared.spa_model.null_fit_source_sha256="e"*64
    with pytest.raises(ValueError,match="fitted-source"):
        binary_single_phewas(prepared.model,g,spa_acquire=acquire,p_filter_cutoff=1)
    assert calls==["enter","exit"]
    prepared.model.null_fit_source_sha256=None
    with pytest.raises(ValueError,match="fitted-source"):
        binary_single_phewas(prepared.model,g,spa_acquire=acquire,p_filter_cutoff=1e-12)


@pytest.mark.parametrize("device", ["cpu","cuda"])
@pytest.mark.parametrize("mixed", [False,True])
def test_temporary_normal_from_fp64_bank_preserves_fitted_products(mixed,device,monkeypatch):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA native TF32 and sparse FP64 source products")
    if device=="cpu":
        # CPU checks stored FP32 fitted formulas; it cannot execute native MMA.
        # CUDA parameters below exercise the actual native route unmodified.
        monkeypatch.setattr("fudan_wgs_toolkit.binary_null.matmul",lambda left,right,*,mode:left@right)
    n=128 if mixed else 64
    x=np.column_stack((np.ones(n),np.linspace(-1,1,n)))
    y=np.tile([0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0],n//16).astype(float)
    rows=np.arange(0,n,2,dtype=np.int64)
    kinship=(SparseKinshipData(np.arange(n).astype(str),np.ones(n),rows,rows+1,
                             np.full(n//2,.5)) if mixed else None)
    fitted=prepare_phewas_model(y,np.arange(n).astype(str),x,kinship=kinship,
        association_mode="tf32",device=device)
    original=fitted.spa_model
    normal=compact_binary_normal_from_fitted_arrays(sample_ids=original.sample_ids,
        covariates=original.x,residual=original.scaled_residuals,
        fitted_probability=original.fitted_probability,
        fixed_effect_covariance=original.fixed_effect_covariance,precision=original.precision,
        null_fit_source_sha256=original.null_fit_source_sha256,
        verified_fit_source_sha256=fitted.model.null_fit_source_sha256,
        has_kinship=original.has_kinship,device=device)
    assert normal.projection_left is None and normal.phenotype is None
    assert not normal.use_spa and normal.matmul_mode=="tf32"
    assert normal.normal_projection_reconstructed and normal.source_matmul_mode=="fp64"
    torch.testing.assert_close(normal.x,fitted.model.x,atol=0,rtol=0)
    torch.testing.assert_close(normal.scaled_residuals,fitted.model.scaled_residuals,atol=0,rtol=0)
    # Dense-family and COO multiplication may round SX differently before
    # casting, so numerical closeness is required instead of universal bits.
    torch.testing.assert_close(normal.precision_x,fitted.model.precision_x,atol=2e-7,rtol=2e-7)
    g=torch.zeros((n,9),dtype=torch.float32,device=device)
    for j in range(9):g[(j*5+np.arange(6))%n,j]=1
    u,v=normal.score_covariance(g)
    expected_u,expected_v=fitted.model.score_covariance(g)
    torch.testing.assert_close(u,expected_u,atol=0,rtol=0)
    torch.testing.assert_close(v,expected_v,atol=2e-6,rtol=2e-6)
    u,variance=normal.individual_score_variance(g)
    expected_u,expected_variance=fitted.model.individual_score_variance(g)
    torch.testing.assert_close(u,expected_u,atol=0,rtol=0)
    torch.testing.assert_close(variance,expected_variance,atol=2e-6,rtol=2e-6)


def immutable_source_arguments(model):
    return dict(sample_ids=model.sample_ids, covariates=model.x,
        residual=model.scaled_residuals, fitted_probability=model.fitted_probability,
        fixed_effect_covariance=model.fixed_effect_covariance, precision=model.precision,
        coefficients=model.coefficients, null_fit_source_sha256=model.null_fit_source_sha256,
        verified_fit_source_sha256=model.null_fit_source_sha256,
        has_kinship=model.has_kinship, iterations=model.iterations, device=model.device)


@pytest.mark.parametrize("mixed", [False, True])
def test_immutable_source_matches_standalone_and_scans_only_at_admission(mixed,monkeypatch):
    prepared,g,maf,annotation=fixture()
    if mixed:
        n=128; rows=np.arange(0,n,2,dtype=np.int64)
        y=np.tile([0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0],n//16).astype(float)
        x=np.column_stack((np.ones(n),np.linspace(-1,1,n)))
        k=SparseKinshipData(np.arange(n).astype(str),np.ones(n),rows,rows+1,np.full(n//2,.5))
        prepared=prepare_phewas_model(y,np.arange(n).astype(str),x,kinship=k,association_mode="fp64")
        g=torch.cat([g,g])
    args=immutable_source_arguments(prepared.spa_model)
    expected_normal=compact_binary_normal_from_fitted_arrays(**args,association_mode="fp64")
    expected_spa=compact_spa_state_from_fitted_arrays(**args)
    expected_single=binary_single_phewas(expected_normal,g,spa_model=expected_spa,p_filter_cutoff=1)
    expected_burden=staar_binary_phewas(expected_normal,g,maf,annotation,
                                      spa_model=expected_spa,p_filter_cutoff=1)
    source=ImmutableFittedBinarySource(**args)
    assert source._tensors[0].data_ptr()==prepared.spa_model.x.data_ptr()
    # Deriving repeated tiles must not repeat the expensive source scan or ID
    # deduplication. Formula and SPA checks run after restoring the spy context.
    with monkeypatch.context() as context:
        def reject_scan(*_args,**_kwargs):
            raise AssertionError("source arrays were scanned again")
        context.setattr(np,"unique",reject_scan)
        context.setattr(torch,"isfinite",reject_scan)
        normal=source.normal(association_mode="fp64")
        spa=source.spa()
        source.normal()
    assert normal.sample_ids is spa.sample_ids is source.sample_ids
    with pytest.raises(ValueError):
        normal.sample_ids.setflags(write=True)
    with pytest.raises(AttributeError,match="immutable"):
        source._binding="f"*64
    for name in ("x","scaled_residuals","fitted_probability","precision_x","fixed_effect_covariance"):
        torch.testing.assert_close(getattr(normal,name),getattr(expected_normal,name),atol=0,rtol=0)
    torch.testing.assert_close(spa.projection_left,expected_spa.projection_left,atol=0,rtol=0)
    actual=binary_single_phewas(normal,g,spa_model=spa,p_filter_cutoff=1)
    torch.testing.assert_close(actual["pvalues"],expected_single["pvalues"],atol=0,rtol=0)
    burden=staar_binary_phewas(normal,g,maf,annotation,spa_model=spa,p_filter_cutoff=1)
    assert burden==expected_burden


@pytest.mark.parametrize("field", ["residual","covariates","precision"])
def test_immutable_source_detects_dense_and_sparse_torch_mutation(field):
    prepared,*_=fixture(); args=immutable_source_arguments(prepared.spa_model)
    if field=="precision":
        args[field]=torch.diag(args[field]).to_sparse_coo().coalesce()
    source=ImmutableFittedBinarySource(**args)
    value=args[field]
    if value.layout==torch.strided:
        value.add_(.01)
    else:
        value.values().add_(.01)
    with pytest.raises(ValueError,match="mutated"):
        source.normal()
    with pytest.raises(ValueError,match="mutated"):
        source.spa()


def test_immutable_source_owns_numpy_arrays_and_rejects_false_binding():
    prepared,*_=fixture(); args=immutable_source_arguments(prepared.spa_model)
    for name in ("covariates","residual","fitted_probability","fixed_effect_covariance",
                 "precision","coefficients"):
        args[name]=args[name].numpy().copy()
    source=ImmutableFittedBinarySource(**args)
    before=source.normal(association_mode="fp64").scaled_residuals.clone()
    args["residual"][:]=99
    torch.testing.assert_close(source.normal(association_mode="fp64").scaled_residuals,before,
                               atol=0,rtol=0)
    with pytest.raises(ValueError,match="fitted-source"):
        ImmutableFittedBinarySource(**dict(args,verified_fit_source_sha256="f"*64))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires native CUDA TF32")
def test_immutable_gpu_source_shares_x_and_preserves_normal_and_spa(monkeypatch):
    n=64; x=np.column_stack((np.ones(n),np.linspace(-1,1,n)))
    y=np.tile([0.,1.,1.,0.,1.,0.,0.,1.],n//8)
    fitted=prepare_phewas_model(y,np.arange(n).astype(str),x,device="cuda",association_mode="tf32")
    args=immutable_source_arguments(fitted.spa_model)
    source=ImmutableFittedBinarySource(**args)
    normal,spa=source.normal(),source.spa()
    assert spa.x.data_ptr()==fitted.spa_model.x.data_ptr()
    g=torch.zeros((n,9),dtype=torch.float32,device="cuda")
    for j in range(9):g[(j*5+np.arange(6))%n,j]=1
    expected=compact_binary_normal_from_fitted_arrays(**args)
    actual_u,actual_v=normal.individual_score_variance(g)
    expected_u,expected_v=expected.individual_score_variance(g)
    torch.testing.assert_close(actual_u,expected_u,atol=0,rtol=0)
    torch.testing.assert_close(actual_v,expected_v,atol=0,rtol=0)
    actual=binary_single_phewas(normal,g,spa_model=spa,p_filter_cutoff=1)
    expected=binary_single_phewas(expected,g,spa_model=compact_spa_state_from_fitted_arrays(**args),p_filter_cutoff=1)
    torch.testing.assert_close(actual["pvalues"],expected["pvalues"],atol=0,rtol=0)
