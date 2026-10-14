"""Bounded resident complete covariance; synthetic contracts are not benchmarks."""
import math
import weakref

import numpy as np
import pytest
import torch

from fudan_wgs_toolkit import tf32
from fudan_wgs_toolkit._gaussian_covariance_resident import (
    panel_width, covariance_new_bytes, resident_gaussian_covariance)
from fudan_wgs_toolkit.phewas_runtime.runtime import (_gaussian_covariance,
                                                    _gene_resident_covariance_new_bytes)
from test_gaussian_covariance_shared import model, genotype, cpu_products


@pytest.mark.parametrize('n,m,budget,expected',[(333001,14567,512*2**20,384),
    (2053,513,4*2053*256,256),(2053,513,4*2053*128,None),
    (23,131,4*23*64,64),(23,32,2**20,None)])
def test_byte_bound_preserves_original_native_geometry(n,m,budget,expected):
    width=panel_width(n,m,maximum_weighted_bytes=budget)
    assert width==expected
    if width is not None:
        assert 4*n*width<=budget
        assert tf32._native_geometry(m,m,n)==tf32._native_geometry(m,width,n)


def test_upload_and_trait_forecast_use_bounded_NEW_weight_and_live_rotation():
    n,m,q=333001,14567,16
    fresh=_gene_resident_covariance_new_bytes(n,m,q,has_blocks=True,maximum_family_rows=2)
    borrowed=_gene_resident_covariance_new_bytes(n,m,q,has_blocks=True,
        maximum_family_rows=2,rotation_ready=True)
    assert fresh-borrowed==4*n*m+16*2*m
    assert fresh < 8*n*m+32*m*m+256*2**20
    assert borrowed>=4*m*m+4*n*384  # Complete C and weighted panel stay mandatory.
    assert fresh+4*n*m>fresh  # A new G upload has its own NEW bytes.


@pytest.mark.parametrize('fortran',[False,True])
@pytest.mark.parametrize('m',[129,131,191])
def test_complete_two_directed_covariance_and_partial_panel_are_retained(cpu_products,fortran,m):
    fitted=model(n=23,shift=.4)
    g=torch.from_numpy(genotype(n=23,m=m,fortran=fortran));before=g.clone()
    expected=_gaussian_covariance(fitted,g)
    guards=[];report={};budget=4*23*64
    actual=resident_gaussian_covariance(fitted,g,maximum_weighted_bytes=budget,
        allocation_guard=lambda size,**phase:guards.append((size,phase['phase'])),report=report)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-5)
    assert torch.equal(actual,actual.T) and actual.shape==(m,m) and torch.equal(g,before)
    assert report['gene_resident_covariance_panels']['directed_column_panels']==math.ceil(m/64)
    assert report['gene_resident_covariance_panels']['maximum_weighted_tensor_bytes']==budget
    assert report['gene_resident_covariance_panels']['zero_padded_output_columns']==math.ceil(m/64)*64-m
    assert all(size>=0 for size,_ in guards)


def test_live_original_rotation_and_cross_are_never_recomputed(cpu_products,monkeypatch):
    fitted=model(n=23);g=torch.from_numpy(genotype(n=23,m=131))
    rotated=fitted.spectrum.rotate(g,matmul_mode='tf32')
    cross=tf32.matmul(fitted.precision_x.T,g,mode='tf32')
    before=(rotated.clone(),cross.clone())
    expected=_gaussian_covariance(fitted,g,rotated=rotated,cross=cross)
    monkeypatch.setattr(fitted.spectrum,'rotate',lambda *a,**k:pytest.fail('borrowed original rotation was rebuilt'))
    actual=resident_gaussian_covariance(fitted,g,rotated=rotated,cross=cross,
        maximum_weighted_bytes=4*23*64)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-5)
    assert torch.equal(rotated,before[0]) and torch.equal(cross,before[1])


def test_refused_new_budget_does_not_allocate_or_rotate(cpu_products,monkeypatch):
    fitted=model(n=23);g=torch.from_numpy(genotype(n=23,m=131));events=[]
    monkeypatch.setattr(fitted.spectrum,'rotate',lambda *a,**k:pytest.fail('preflight refusal rotated'))
    monkeypatch.setattr(torch,'empty',lambda *a,**k:pytest.fail('preflight refusal allocated'))
    def refuse(size,*,phase):
        events.append((size,phase));raise MemoryError('explicit CPU contract budget')
    with pytest.raises(MemoryError) as captured:
        resident_gaussian_covariance(fitted,g,maximum_weighted_bytes=4*23*64,allocation_guard=refuse)
    assert len(events)==1
    assert captured.value.gaussian_covariance_failure['complete_covariance_returned'] is False
    assert captured.value.gaussian_covariance_failure['variants_removed'] is False


def test_actual_product_error_releases_owned_weight_and_C_with_retained_exception(cpu_products,monkeypatch):
    fitted=model(n=23);g=torch.from_numpy(genotype(n=23,m=131));owned=[]
    original_empty=torch.empty;original_product=tf32.matmul
    def allocate(*a,**k):
        value=original_empty(*a,**k);owned.append(weakref.ref(value));return value
    def fail(a,b,*,mode):
        if a.shape[0]==131 and b.shape[1]==64:
            raise torch.cuda.OutOfMemoryError('actual weighted product fixture failure')
        return original_product(a,b,mode=mode)
    monkeypatch.setattr(torch,'empty',allocate);monkeypatch.setattr(tf32,'matmul',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError) as captured:
        resident_gaussian_covariance(fitted,g,maximum_weighted_bytes=4*23*64)
    assert owned and all(reference() is None for reference in owned)
    assert captured.value.gaussian_covariance_failure['precision_changed'] is False


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA covariance needs CUDA')
@pytest.mark.parametrize('m,fortran',[(513,False),(513,True),(734,False),(734,True)])
def test_actual_cuda_complete_covariance_matches_original_MMA_with_singleton_tail(m,fortran):
    fitted=model(n=2053,shift=.4,device='cuda:0')
    g=torch.as_tensor(genotype(n=2053,m=m,fortran=fortran),device='cuda:0')
    expected=_gaussian_covariance(fitted,g)
    report={};budget=4*2053*256
    actual=resident_gaussian_covariance(fitted,g,maximum_weighted_bytes=budget,report=report)
    torch.cuda.synchronize()
    # Original whole-R/direct TF32 geometry, both entries and FP32 symmetrization.
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.isfinite(actual).all() and torch.equal(actual,actual.T)
    assert report['gene_resident_covariance_panels']['maximum_weighted_tensor_bytes']<=budget
