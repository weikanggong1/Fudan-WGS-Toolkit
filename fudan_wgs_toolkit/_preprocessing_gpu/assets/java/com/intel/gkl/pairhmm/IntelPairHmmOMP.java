package com.intel.gkl.pairhmm;

/** Original OMP provider is preserved inside the isolated original-GKL loader. */
public final class IntelPairHmmOMP extends IntelPairHmm {
    public IntelPairHmmOMP() { useOmp=true; }
}
