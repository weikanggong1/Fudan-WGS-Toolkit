"""Second candidate: fuse FP64 elementwise derivatives, keep Torch sum geometry."""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def _elements(X,MU,G,L,A,B,N:tl.constexpr,M:tl.constexpr,GS0:tl.constexpr,GS1:tl.constexpr,
              MS0:tl.constexpr,MS1:tl.constexpr,LS0:tl.constexpr,LS1:tl.constexpr,
              SHARED:tl.constexpr,XS:tl.constexpr,BLOCK:tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=offset<N*M
    row=offset//M;col=offset%M
    mu_col=tl.full((BLOCK,),0,tl.int32) if SHARED else col
    x=tl.load(X+col*XS,valid,other=0.)
    g=tl.load(G+row*GS0+col*GS1,valid,other=0.)
    mu=tl.load(MU+row*MS0+mu_col*MS1,valid,other=0.)
    log_odds=tl.load(L+row*LS0+mu_col*LS1,valid,other=0.)
    tilted=log_odds+g*x
    one=tl.full((BLOCK,),1.,tl.float64)
    probability=libdevice.div_rn(one,one+libdevice.exp(-tilted))
    control=libdevice.div_rn(one,one+libdevice.exp(tilted))
    first=(probability-mu)*g
    second=(probability*control)*(g*g)
    tl.store(A+offset,first,valid)
    tl.store(B+offset,second,valid)


def derivatives(x,mu,g,q,log_odds):
    n,m=g.shape
    first=torch.empty((n,m),dtype=torch.float64,device=g.device)
    second=torch.empty_like(first)
    _elements[(triton.cdiv(n*m,256),)](x,mu,g,log_odds,first,second,n,m,*g.stride(),
        *mu.stride(),*log_odds.stride(),mu.shape[1]==1,x.stride(0),256,num_warps=4,enable_fp_fusion=False)
    return first.sum(0)-q,second.sum(0)
