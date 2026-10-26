# SPDX-License-Identifier: GPL-3.0-only
# Exact _cct_tensor reference from upstream commit d64de4968feb79c0dd8aa990ed531b01fcde2ca4.
# Loaded with the statistics module globals by test_runtime_optimization.py.
def _cct_tensor(pvalues, weights=None, *, internal: bool = False, sync_light: bool = False, _normal_transform=None) -> torch.Tensor:
    p = _double(pvalues).reshape(-1)
    if sync_light:
        from ._statistics_sync import host_flags
        if not p.numel():
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        finite, out_of_range, has_zero, has_one = host_flags(
            torch.isfinite(p).all(), ((p < 0) | (p > 1)).any(), (p == 0).any(), (p == 1).any())
        if not finite or out_of_range:
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        if not internal:
            if has_zero and has_one:
                raise ValueError("cannot combine both exact 0 and exact 1 pvalues")
            if has_zero:
                return p.new_tensor(0.)
            if has_one:
                warnings.warn("there are pvalues that are exactly 1", RuntimeWarning, stacklevel=3)
                return p.new_tensor(1.)
    else:
        if not p.numel() or not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any()):
            raise ValueError("pvalues must be a nonempty finite vector in [0, 1]")
        if not internal:
            if bool((p == 0).any()) and bool((p == 1).any()):
                raise ValueError("cannot combine both exact 0 and exact 1 pvalues")
            if bool((p == 0).any()):
                return p.new_tensor(0.)
            if bool((p == 1).any()):
                warnings.warn("there are pvalues that are exactly 1", RuntimeWarning, stacklevel=3)
                return p.new_tensor(1.)
    if weights is None:
        w = torch.full_like(p, 1 / p.numel())
    else:
        w = _double(weights, device=p.device).reshape(-1)
        if sync_light:
            if w.shape != p.shape:
                raise ValueError("weights must be finite, nonnegative and match pvalues")
            mass = w.sum()
            finite, negative, mass_finite, mass_nonpositive = host_flags(
                torch.isfinite(w).all(), (w < 0).any(), torch.isfinite(mass), mass <= 0)
            if not finite or negative:
                raise ValueError("weights must be finite, nonnegative and match pvalues")
            if not mass_finite or mass_nonpositive:
                raise DegenerateTestError("CCT requires a positive finite weight sum")
        else:
            if w.shape != p.shape or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
                raise ValueError("weights must be finite, nonnegative and match pvalues")
            mass = w.sum()
            if not bool(torch.isfinite(mass)) or bool(mass <= 0):
                raise DegenerateTestError("CCT requires a positive finite weight sum")
        w = w / mass
    small = p < 1e-16
    statistic = (w[small] / p[small] / math.pi).sum()
    normal_transform = (torch.tan((0.5 - p[~small]) * math.pi)
                        if _normal_transform is None else _normal_transform[~small])
    statistic = statistic + (w[~small] * normal_transform).sum()
    if sync_light:
        is_nan, large, nonpositive = host_flags(torch.isnan(statistic), statistic > 1e15, statistic <= 0)
        if is_nan:
            raise DegenerateTestError("CCT statistic is undefined")
        if large:
            return (1 / statistic) / math.pi
        tail = _cauchy_sf_tensor(statistic)
        return tail if internal or nonpositive else 1 - (1 - tail)
    if bool(torch.isnan(statistic)):
        raise DegenerateTestError("CCT statistic is undefined")
    if bool(statistic > 1e15):
        return (1 / statistic) / math.pi
    tail = _cauchy_sf_tensor(statistic)
    return tail if internal or bool(statistic <= 0) else 1 - (1 - tail)
