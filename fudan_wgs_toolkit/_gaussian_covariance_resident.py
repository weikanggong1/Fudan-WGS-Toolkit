"""Complete resident covariance without a second whole weighted genotype.

The original whole-mask rotation and fitted covariate cross are retained.
Only the output columns of their two directed products are partitioned. Last
panels are padded with zero output columns to retain the original TF32 MMA
geometry, including masks whose last panel contains one actual variant.
"""
import traceback
import torch

MAXIMUM_WEIGHTED_BYTES = 512*2**20


def panel_width(n, m, *, maximum_weighted_bytes=MAXIMUM_WEIGHTED_BYTES):
    if (type(n) is not int or type(m) is not int or n < 1 or m < 1
            or type(maximum_weighted_bytes) is not int or maximum_weighted_bytes < 1):
        raise ValueError('positive dimensions and weighted byte budget required')
    # Native large outputs use BM64/BN128; small outputs use BM32/BN64.
    # Do not turn a mature MMA product into GEMV or change its K reduction.
    large = m >= 256 and n >= 1024
    unit, minimum = (128, 256) if large else (64, 64)
    capacity = maximum_weighted_bytes//(4*n)
    width = min(512, capacity//unit*unit)
    return width if minimum <= width < m else None


def covariance_new_bytes(n, m, q, *, rotation_new_bytes=0,
                         maximum_weighted_bytes=MAXIMUM_WEIGHTED_BYTES):
    """NEW output, panel, projection and rotation peak; inputs already live."""
    width = panel_width(n,m,maximum_weighted_bytes=maximum_weighted_bytes)
    if width is None:
        raise MemoryError('weighted panel budget cannot retain original native geometry')
    # C; inverse-weighted panel; zero-padded cross; whole original projection;
    # directed Gram/projection/subtraction columns; block symmetry temporaries.
    return (rotation_new_bytes+4*m*m+4*n*width+4*q*width+8*m*q
            +12*m*width+8*width*width+256*2**20)


def resident_gaussian_covariance(model, genotype, *, rotated=None, cross=None,
                                allocation_guard=None, report=None,
                                maximum_weighted_bytes=MAXIMUM_WEIGHTED_BYTES):
    """Evaluate every original FP32 weighted product and complete covariance.

    This preflight is not an allocator guarantee. A caller may retry a real
    admission/allocation failure through its mature bounded host path. No
    exception is suppressed here and no sample, variant or precision changes.
    ``allocation_guard(required, phase=...)`` replaces the caller's NEW-byte
    reservation as rotation, cross and output become resident.
    """
    from .tf32 import matmul, _native_geometry, _estimated_product_workspace
    n,m=map(int,genotype.shape);q=int(model.precision_x.shape[1])
    width=panel_width(n,m,maximum_weighted_bytes=maximum_weighted_bytes)
    if width is None:
        raise MemoryError('weighted panel budget cannot retain original native geometry')
    if (genotype.dtype!=torch.float32 or model.inverse_variance.dtype!=torch.float32
            or model.fixed_effect_covariance.dtype!=torch.float32):
        raise ValueError('resident fitted covariance requires original FP32 storage')
    if _native_geometry(m,m,n)!=_native_geometry(m,width,n):
        raise ValueError('weighted panel would change original TF32 geometry')
    covariance=projected=weighted=cross_panel=gram=correction=symmetric=None
    panel_calls=0; padded_columns=0
    cross_new=(0 if cross is not None else _estimated_product_workspace(
        (q,n),(n,m),mode='tf32',left_dtype=model.precision_x.dtype,
        right_dtype=genotype.dtype))
    largest_family=max((len(rows) for rows,_ in model.spectrum.blocks),default=0)
    rotation_new=(4*n*m+16*largest_family*m if rotated is None and model.spectrum.blocks else 0)
    phase='before_rotation';maximum_new=0
    if report is not None:
        value=report.setdefault('gene_resident_covariance_panels',{})
        value['attempts']=value.get('attempts',0)+1
        value['last_geometry']=dict(n=n,m=m,weighted_panel_columns=width,
            maximum_family_rows=largest_family,weighted_tensor_bytes=4*n*width,
            original_whole_weighted_tensor_bytes=4*n*m)
    def guard(required, phase):
        nonlocal maximum_new
        maximum_new=max(maximum_new,required)
        if report is not None:
            report['gene_resident_covariance_panels']['last_new_workspace_phase']=phase
            report['gene_resident_covariance_panels']['maximum_new_workspace_bytes']=max(
                report['gene_resident_covariance_panels'].get('maximum_new_workspace_bytes',0),required)
        if allocation_guard is not None:
            allocation_guard(required,phase=phase)
    try:
        guard(covariance_new_bytes(n,m,q,rotation_new_bytes=rotation_new,
            maximum_weighted_bytes=maximum_weighted_bytes)+cross_new,'gene_resident_covariance_rotation')
        if rotated is None:
            rotated=model.spectrum.rotate(genotype,matmul_mode='tf32')
        phase='cross'
        # R is now allocated, including a caller-borrowed original rotation.
        guard(covariance_new_bytes(n,m,q,maximum_weighted_bytes=maximum_weighted_bytes)
            +cross_new,'gene_resident_covariance_cross')
        if cross is None:
            cross=matmul(model.precision_x.T,genotype,mode='tf32')
        phase='output'
        guard(covariance_new_bytes(n,m,q,maximum_weighted_bytes=maximum_weighted_bytes),
            'gene_resident_covariance_output')
        covariance=torch.empty((m,m),dtype=torch.float32,device=genotype.device)
        projected=matmul(cross.T,model.fixed_effect_covariance,mode='tf32')
        future=4*n*width+4*q*width+12*m*width+8*width*width+256*2**20
        guard(future,'gene_resident_covariance_weighted_panel')
        weighted=torch.empty((n,width),dtype=torch.float32,device=genotype.device)
        cross_panel=torch.empty((q,width),dtype=torch.float32,device=genotype.device)
        phase='weighted_directed_panels'
        for begin in range(0,m,width):
            end=min(begin+width,m);count=end-begin
            torch.mul(model.inverse_variance[:,None],rotated[:,begin:end],out=weighted[:,:count])
            cross_panel[:,:count].copy_(cross[:,begin:end])
            if count < width:
                weighted[:,count:].zero_();cross_panel[:,count:].zero_()
                padded_columns+=width-count
            gram=matmul(rotated.T,weighted,mode='tf32')
            correction=matmul(projected,cross_panel,mode='tf32')
            gram.sub_(correction)
            covariance[:,begin:end].copy_(gram[:,:count])
            gram=correction=None;panel_calls+=1
        # Both original directed entries are present before overwriting a pair.
        # Keep the original FP32 (V+V.T)/2 order; no triangular approximation.
        phase='complete_original_symmetrization'
        for begin in range(0,m,width):
            end=min(begin+width,m)
            for other in range(begin,m,width):
                stop=min(other+width,m)
                symmetric=(covariance[begin:end,other:stop]
                    +covariance[other:stop,begin:end].T)/2
                covariance[begin:end,other:stop].copy_(symmetric)
                if other != begin:
                    covariance[other:stop,begin:end].copy_(symmetric.T)
                symmetric=None
        if report is not None:
            value=report.setdefault('gene_resident_covariance_panels',{})
            value['calls']=value.get('calls',0)+1
            value['directed_column_panels']=value.get('directed_column_panels',0)+panel_calls
            value['zero_padded_output_columns']=value.get('zero_padded_output_columns',0)+padded_columns
            value['maximum_weighted_tensor_bytes']=max(value.get('maximum_weighted_tensor_bytes',0),4*n*width)
            value['maximum_whole_weighted_tensor_avoided_bytes']=max(value.get('maximum_whole_weighted_tensor_avoided_bytes',0),4*n*m)
            value.update(complete_covariance=True,original_whole_rotation=True,
                original_whole_cross=True,own_fitted_precision=True,
                original_native_geometry=True,weighted_byte_limit=maximum_weighted_bytes,
                probability_definition_changed=False)
        return covariance
    except BaseException as error:
        error.gaussian_covariance_failure=dict(route='resident_complete_weighted_column_panels',
            error_type=type(error).__name__,phase=phase,n=n,m=m,
            weighted_panel_columns=width,weighted_tensor_bytes=4*n*width,
            maximum_family_rows=largest_family,maximum_estimated_new_workspace_bytes=maximum_new,
            complete_covariance_returned=False,variants_removed=False,precision_changed=False)
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        # A retained caller traceback must not own R, G or an unreturned C.
        covariance=projected=weighted=cross_panel=gram=correction=symmetric=None
        rotated=cross=model=genotype=allocation_guard=report=guard=None
