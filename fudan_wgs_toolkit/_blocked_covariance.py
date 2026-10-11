"""Mask-local, byte-bounded preparation reuse for fitted kinship panels.

Only preparation is cached. Every directed covariance product retains the
original native TF32 panel shape and each fitted model's own precision. This
first path does not share a covariance or rotation across fitted models.
"""
from contextlib import contextmanager, nullcontext
import torch


def _storage_bytes(values):
    """Count actual distinct storage; diagonal rotation can alias raw G."""
    seen, total = set(), 0
    for value in values:
        if value is None:
            continue
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


class _PanelPreparation:
    def __init__(self, model, host, broker, width, capacity_bytes, profiler=None):
        self.model, self.host, self.broker = model, host, broker
        self.width, self.capacity_bytes, self.profiler = width, capacity_bytes, profiler
        self.panels, self.projections, self.pins = {}, {}, {}
        self.panel_bytes = self.projection_bytes = 0
        self.metrics = dict(capacity_bytes=capacity_bytes, peak_storage_bytes=0,
                            panel_preparations=0, panel_hits=0, panel_evictions=0,
                            projection_preparations=0, projection_hits=0,
                            projection_evictions=0, pressure_releases=0)

    def measure(self, phase):
        return (self.profiler.measure(phase, gpu=True)
                if self.profiler is not None else nullcontext())

    def _peak(self):
        self.metrics['peak_storage_bytes'] = max(self.metrics['peak_storage_bytes'],
                                               self.panel_bytes+self.projection_bytes)

    def _remove_panel(self, begin):
        _, size = self.panels.pop(begin)
        self.panel_bytes -= size
        self.metrics['panel_evictions'] += 1

    def release_unpinned(self):
        """Retire optional owners without crediting borrowed live tensors."""
        released = 0
        for begin in tuple(self.panels):
            if not self.pins.get(begin):
                released += self.panels[begin][1]
                self._remove_panel(begin)
        for begin in tuple(self.projections):
            if not self.pins.get(begin):
                _, size = self.projections.pop(begin)
                self.projection_bytes -= size
                self.metrics['projection_evictions'] += 1
                released += size
        self.metrics['pressure_releases'] += 1
        return released

    def _guard(self, new_bytes):
        try:
            self.broker._guard(new_bytes)
        except MemoryError:
            self.release_unpinned()
            self.broker._guard(new_bytes)

    def _admit(self, begin, payload, size):
        # Higher right panels are reused by more left panels. Ordinary LRU
        # would thrash during the increasing nested traversal.
        while self.panel_bytes+self.projection_bytes+size > self.capacity_bytes:
            victims = [key for key in self.panels if not self.pins.get(key)]
            if not victims:
                return
            victim = min(victims)
            if begin < victim:
                return
            self._remove_panel(victim)
        if size <= self.capacity_bytes:
            self.panels[begin] = (payload, size)
            self.panel_bytes += size
            self._peak()

    @contextmanager
    def borrow(self, begin, end):
        raw = rotated = cross = projected = payload = None
        self.pins[begin] = self.pins.get(begin, 0)+1
        try:
            if begin in self.panels:
                payload = self.panels[begin][0]
                raw, rotated = payload
                self.metrics['panel_hits'] += 1
            else:
                n, w = self.host.shape[0], end-begin
                # Raw, rotation clone, possible indexed rotation temporaries
                # and a weighted panel remain covered before materialization.
                self._guard(20*n*w+256*2**20)
                with self.measure('gene_panel_h2d'):
                    raw = torch.as_tensor(self.host[:, begin:end], dtype=torch.float32,
                                          device=self.model.device)
                    if not raw.is_cuda:
                        raw = raw.clone()  # CPU contract oracle owns this panel.
                with self.measure('gene_panel_rotation'):
                    rotated = self.model.spectrum.rotate(raw, matmul_mode='tf32')
                payload = (raw, rotated)
                self.metrics['panel_preparations'] += 1
                self._admit(begin, payload, _storage_bytes(payload))
            if begin in self.projections:
                (cross, projected), _ = self.projections[begin]
                self.metrics['projection_hits'] += 1
            else:
                from .tf32 import matmul
                self._guard(12*self.model.x.shape[1]*(end-begin)+256*2**20)
                with self.measure('gene_panel_projection'):
                    cross = matmul(self.model.precision_x.T, raw, mode='tf32')
                    projected = matmul(cross.T, self.model.fixed_effect_covariance,
                                       mode='tf32')
                size = _storage_bytes((cross, projected))
                while self.panel_bytes+self.projection_bytes+size > self.capacity_bytes:
                    victims = [key for key in self.panels if not self.pins.get(key)]
                    if not victims:
                        break
                    self._remove_panel(min(victims))
                # Both payload count and bytes are bounded. Active borrowers
                # retain their own small projection if caching cannot fit.
                if (len(self.projections) < 256 and
                        self.panel_bytes+self.projection_bytes+size <= self.capacity_bytes):
                    self.projections[begin] = ((cross, projected), size)
                    self.projection_bytes += size
                    self._peak()
                self.metrics['projection_preparations'] += 1
            yield raw, rotated, cross, projected
        finally:
            self.pins[begin] -= 1
            if not self.pins[begin]:
                del self.pins[begin]
            raw = rotated = cross = projected = payload = None

    def close(self):
        self.panels.clear()
        self.projections.clear()
        self.pins.clear()
        self.panel_bytes = self.projection_bytes = 0
        self.host = self.model = self.broker = self.profiler = None


def blocked_gaussian_covariance(model, host, broker, *, variant_tile_size=512,
                                profiler=None, report=None, capacity_bytes=None):
    """Reuse preparation within one model/mask; keep native pair mathematics.

    The default optional cache is at most 8 GiB and 15% of currently available
    CUDA space after the minimum temporary allowance. It is subordinate to
    the caller's total budget and its current live workspace admission.
    Clearing this cache never grants artificial allocator/free-memory credit.
    """
    from .tf32 import matmul
    if type(variant_tile_size) is not int or variant_tile_size < 1:
        raise ValueError('variant_tile_size must be a positive integer')
    n, m = host.shape
    if m < 1:
        raise ValueError('a covariance mask must contain a variant')
    width = min(m, variant_tile_size)
    scratch = 20*n*width+16*width**2+256*2**20
    broker._guard(4*m*m+scratch)
    covariance = cache = block = reverse = weighted = None
    left = left_rotated = left_cross = projected_left = None
    right = right_rotated = right_cross = projected_right = None
    try:
        covariance = torch.empty((m, m), dtype=torch.float32, device=model.device)
        if capacity_bytes is None:
            live = getattr(broker, '_live_available_bytes', None)
            capacity_bytes = (min(8*2**30, int(max(0, live()-scratch)*.15))
                              if callable(live) else 0)
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError('capacity_bytes must be a nonnegative integer')
        cache = _PanelPreparation(model, host, broker, width, capacity_bytes, profiler)
        release_scope = getattr(broker, 'optional_device_cache', None)
        scope = release_scope(cache.release_unpinned) if callable(release_scope) else nullcontext()
        with scope:
            for begin in range(0, m, width):
                end = min(begin+width, m)
                with cache.borrow(begin, end) as left_values:
                    left, left_rotated, left_cross, projected_left = left_values
                    for other in range(begin, m, width):
                        stop = min(other+width, m)
                        right_scope = (nullcontext(left_values) if other == begin
                                       else cache.borrow(other, stop))
                        with right_scope as right_values:
                            right, right_rotated, right_cross, projected_right = right_values
                            cache._guard(8*n*width+16*width**2+256*2**20)
                            with cache.measure('gene_panel_covariance_products'):
                                weighted = model.inverse_variance[:, None]*right_rotated
                                block = (matmul(left_rotated.T, weighted, mode='tf32')
                                         -matmul(projected_left, right_cross, mode='tf32'))
                                weighted = None
                                if other == begin:
                                    block = (block+block.T)/2
                                else:
                                    weighted = model.inverse_variance[:, None]*left_rotated
                                    reverse = (matmul(right_rotated.T, weighted, mode='tf32')
                                               -matmul(projected_right, left_cross, mode='tf32'))
                                    weighted = None
                                    block = (block+reverse.T)/2
                                    reverse = None
                                covariance[begin:end, other:stop].copy_(block)
                                if other != begin:
                                    covariance[other:stop, begin:end].copy_(block.T)
                            right_values = right = right_rotated = right_cross = projected_right = block = None
                        right_scope = None
                    left_values = left = left_rotated = left_cross = projected_left = None
        return covariance
    finally:
        if cache is not None:
            metrics = dict(cache.metrics)
            cache.close()
            metrics.update(storage_bytes_after_close=cache.panel_bytes+cache.projection_bytes,
                           pinned_panels_after_close=len(cache.pins),
                           model_scopes=1, covariance_shared=False)
            if report is not None:
                summary = report.setdefault('gene_panel_preparation_cache', {})
                for key, value in metrics.items():
                    if isinstance(value, bool):
                        summary[key] = value
                    elif key in ('capacity_bytes', 'peak_storage_bytes'):
                        summary[key] = max(summary.get(key, 0), value)
                    else:
                        summary[key] = summary.get(key, 0)+value
        cache = covariance = block = reverse = weighted = None
        left = left_rotated = left_cross = projected_left = None
        right = right_rotated = right_cross = projected_right = None
        left_values = right_values = scope = release_scope = right_scope = None
        host = model = broker = profiler = report = None
