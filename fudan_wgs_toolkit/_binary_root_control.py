"""Experimental FP64 SPA control and elementwise derivative fusion; reductions stay unchanged.

The candidate fuses scalar state updates and elementwise FP64 derivatives.
It retains reference active-column compaction and exact Torch reduction shapes.
No approximate reciprocal, reassociation, fused multiply-add, or tolerance change.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _bracket_update(INDEX, FIRST, Q, END, PENDING, BRACKETED, COUNTS,
                    K:tl.constexpr, BLOCK:tl.constexpr):
    lane=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    live=lane<K
    index=tl.load(INDEX+lane,live,other=0)
    first=tl.load(FIRST+lane,live,other=0.)
    q=tl.load(Q+index,live,other=0.)
    end=tl.load(END+index,live,other=0.)
    found=(tl.abs(first)<float('inf')) & tl.where(q>0.,first>=0.,first<=0.)
    count=tl.load(COUNTS+index,live,other=0)
    tl.store(COUNTS+index,count+1,live)
    tl.store(BRACKETED+index,found,live)
    tl.store(PENDING+index,~found,live)
    tl.store(END+index,tl.where(found,end,end*2.),live)

@triton.jit
def _root_update(INDEX,FIRST,CURVATURE,CURRENT,LO,HI,TOL,ROOT,RESIDUAL,
                 CONVERGED,COUNTS,USED_BISECTION,ACTIVE,
                 K:tl.constexpr,NEWTON:tl.constexpr,BLOCK:tl.constexpr):
    lane=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    live=lane<K
    index=tl.load(INDEX+lane,live,other=0)
    first=tl.load(FIRST+lane,live,other=0.)
    curvature=tl.load(CURVATURE+lane,live,other=1.)
    current=tl.load(CURRENT+index,live,other=0.)
    lower=tl.load(LO+index,live,other=0.)
    upper=tl.load(HI+index,live,other=0.)
    tolerance=tl.load(TOL+index,live,other=0.)
    finite=tl.abs(first)<float('inf')
    done=finite & (tl.abs(first)<=tolerance)
    lower=tl.where(first<0.,current,lower)
    upper=tl.where(first>=0.,current,upper)
    if NEWTON:
        delta=tl.inline_asm_elementwise("div.rn.f64 $0, $1, $2;",
             constraints="=d,d,d",args=[first,curvature],dtype=tl.float64,
             is_pure=True,pack=1)
        candidate=current-delta
    else:
        candidate=tl.full((BLOCK,),float('nan'),tl.float64)
    take=(tl.abs(candidate)<float('inf')) & (curvature>0.) & (candidate>lower) & (candidate<upper)
    midpoint=lower+(upper-lower)/2.
    update=tl.where(take,candidate,midpoint)
    continues=(~done) & finite & (update!=current)
    count=tl.load(COUNTS+index,live,other=0)
    used=tl.load(USED_BISECTION+index,live,other=False)
    tl.store(COUNTS+index,count+1,live)
    tl.store(ROOT+index,current,live)
    tl.store(RESIDUAL+index,tl.abs(first),live)
    tl.store(CONVERGED+index,done,live)
    tl.store(LO+index,lower,live)
    tl.store(HI+index,upper,live)
    tl.store(USED_BISECTION+index,used | ((~take) & (~done)),live)
    tl.store(ACTIVE+index,continues,live)
    tl.store(CURRENT+index,update,live)


def safeguarded_root(mu,g,q,tol,max_iter,*,newton=True):
    from .binary import _SPARootResult,_score_support,_probability_columns
    from ._binary_root_derivatives import derivatives as _derivatives_from_log_odds
    zero=torch.zeros_like(q)
    flags=torch.zeros_like(q,dtype=torch.bool)
    counts=torch.zeros_like(q,dtype=torch.int64)
    result=_SPARootResult(zero.clone(),flags.clone(),torch.full_like(q,float('inf')),
                         counts.clone(),flags.clone(),flags.clone(),counts.clone())
    log_odds=torch.log(mu)-torch.log1p(-mu)
    variance=(mu*(1-mu)*g.square()).sum(0)
    lower,upper=_score_support(mu,g)
    valid=(variance>0)&torch.isfinite(variance)&(q>lower)&(q<upper)
    centered=valid&(q==0)
    result.converged[centered]=True
    result.residual[centered]=0
    residual_tol=min(tol,1e-10)*torch.maximum(variance.sqrt(),q.abs())
    step=(q/variance).abs().clamp(min=1e-3,max=8.)
    end=torch.where(q>0,step,-step)
    pending=valid&~centered
    bracketed=flags.clone()
    for _ in range(128):
        index=torch.nonzero(pending,as_tuple=True)[0]
        if not len(index):break
        first,_=_derivatives_from_log_odds(end[index],_probability_columns(mu,index),
             g[:,index],q[index],_probability_columns(log_odds,index))
        _bracket_update[(triton.cdiv(len(index),128),)](index,first,q,end,pending,bracketed,
            result.bracket_iterations,len(index),128,num_warps=4,enable_fp_fusion=False)
    lo,hi=torch.minimum(end,zero),torch.maximum(end,zero)
    current=q/variance
    ordinary=torch.isfinite(current)&(current>lo)&(current<hi)
    current=torch.where(ordinary,current,lo+(hi-lo)/2)
    result.used_bisection=bracketed&~ordinary
    active=bracketed.clone()
    for _ in range(max_iter):
        index=torch.nonzero(active,as_tuple=True)[0]
        if not len(index):break
        first,curvature=_derivatives_from_log_odds(current[index],_probability_columns(mu,index),
             g[:,index],q[index],_probability_columns(log_odds,index))
        _root_update[(triton.cdiv(len(index),128),)](index,first,curvature,current,lo,hi,residual_tol,
            result.root,result.residual,result.converged,result.iterations,result.used_bisection,active,
            len(index),newton,128,num_warps=4,enable_fp_fusion=False)
    result.iteration_limit=active
    return result
