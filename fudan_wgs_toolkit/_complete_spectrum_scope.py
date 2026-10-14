"""Explicit, bounded complete-spectrum optimizations; no approximate spectrum.

Batching changes only how complete H32 matrices reach the same eigensolver.
Reuse is confined to one caller-owned immutable covariance lifetime. It never
identifies different fitted models by shape, sample count or proportionality.
"""
from contextlib import contextmanager
import copy
import threading
import weakref

import torch

_LOCAL = threading.local()
_METRICS = dict(batch_policy_calls=0, exact_relation_calls=0,
    exact_relation_host_transfers=0, max_relation_workspace_bytes=0,
    relation_budget_declines=0, reuse_scopes=0, reuse_hits=0,
    reuse_misses=0, reuse_captures=0, reuse_budget_declines=0,
    reuse_unversioned_declines=0, reuse_input_changes=0,
    max_retained_spectrum_weight_bytes=0, live_retained_bytes=0,
    closed_reuse_scopes=0)


@contextmanager
def complete_spectrum_batch_scope(*, maximum_variants=256, weight_batch_size=32,
                                  relation_workspace_bytes=256 * 2**10):
    """Opt in to complete weight batching for an actual, bounded matrix size.

Larger matrices retain the ordinary complete sequential route. The relation
budget bounds boolean comparison tiles, rather than an M-by-J-by-J tensor.
The process/live GPU admission remains the final matrix-batch bound.
"""
    for name, value, minimum in (("maximum_variants", maximum_variants, 1),
            ("weight_batch_size", weight_batch_size, 1),
            ("relation_workspace_bytes", relation_workspace_bytes, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(name + " must be an integer at least " + str(minimum))
    previous = getattr(_LOCAL, 'batch', None)
    policy = dict(maximum_variants=maximum_variants,
                  weight_batch_size=weight_batch_size,
                  relation_workspace_bytes=relation_workspace_bytes)
    _LOCAL.batch = policy
    try:
        yield policy.copy()
    finally:
        _LOCAL.batch = previous


def batch_policy(number_variants):
    policy = getattr(_LOCAL, 'batch', None)
    if policy is None or number_variants > policy['maximum_variants']:
        return None
    _METRICS['batch_policy_calls'] += 1
    return policy


def exact_weight_mapping(weights, budget):
    """One host transfer of bounded, bit-exact J-by-J relation flags."""
    bits = relations = block = flags = host = None
    try:
        if (not isinstance(weights, torch.Tensor) or weights.dtype != torch.float32
                or weights.ndim != 2 or min(weights.shape) < 1):
            raise ValueError('complete weight relations require a nonempty FP32 matrix')
        m, j = weights.shape
        if j == 1:
            return [0], [0]
        cells = j * j
        rows = min(64, m, max(0, budget // cells - 2))
        if rows < 1:
            _METRICS['relation_budget_declines'] += 1
            return None
        required = (rows + 2) * cells
        if weights.is_cuda:
            from ._fit_structure_cache import guard_cuda_workspace
            try:
                guard_cuda_workspace(required, weights.device,
                                     context='complete spectrum exact-weight relation tile')
            except MemoryError:
                # Relation batching is optional. The ordinary per-weight route
                # still performs its own original full-matrix admission.
                _METRICS['relation_budget_declines'] += 1
                return None
        _METRICS['exact_relation_calls'] += 1
        _METRICS['max_relation_workspace_bytes'] = max(
            _METRICS['max_relation_workspace_bytes'], required)
        bits = weights.view(torch.int32)
        relations = torch.ones((j, j), dtype=torch.bool, device=weights.device)
        for start in range(0, m, rows):
            block = bits[start:start + rows]
            flags = (block[:, :, None] == block[:, None, :]).all(dim=0)
            relations.logical_and_(flags)
            flags = block = None
        host = relations.to(device='cpu').tolist()
        _METRICS['exact_relation_host_transfers'] += int(weights.is_cuda)
        representatives, mapping = [], []
        for index in range(j):
            duplicate = next((previous for previous in representatives
                              if host[index][previous]), None)
            if duplicate is None:
                representatives.append(index)
                duplicate = index
            mapping.append(duplicate)
        return representatives, mapping
    finally:
        weights = bits = relations = block = flags = host = None


def _signature(tensor):
    try:
        return (tensor._version, tensor.data_ptr(), tuple(tensor.shape),
                tuple(tensor.stride()), tensor.storage_offset(),
                str(tensor.dtype), str(tensor.device))
    except RuntimeError:  # Inference tensors have no mutation counter.
        return None


class _ReuseScope:
    def __init__(self, limit):
        self.limit = limit
        self.matrix = self.weights = self.result = None
        self.matrix_signature = self.weight_signature = self.result_signature = None
        self.binding = self.current_binding = None
        self.retained_bytes = 0

    def clear(self):
        _METRICS['live_retained_bytes'] -= self.retained_bytes
        self.retained_bytes = 0
        self.matrix = self.weights = self.result = None
        self.matrix_signature = self.weight_signature = self.result_signature = None
        self.binding = None

    def register_transform(self, original, transformed, selection):
        # Installed only after association_test's unchanged FP32 rare subset
        # and symmetric-mean operations. All matrix references remain weak.
        if (original.dtype != torch.float32 or transformed.dtype != torch.float32
                or selection.dtype != torch.bool or selection.ndim != 1
                or original.ndim != 2 or original.shape[0] != original.shape[1]
                or transformed.ndim != 2 or transformed.shape[0] != transformed.shape[1]
                or len(selection) != original.shape[0]
                or transformed.device != original.device or selection.device != original.device):
            raise ValueError('complete covariance transform needs aligned FP32 C and its actual boolean rare mask')
        self.current_binding = (weakref.ref(original), _signature(original),
            weakref.ref(transformed), _signature(transformed), selection.detach(),
            _signature(selection))

    def same_transform(self, covariance):
        previous, current = self.binding, self.current_binding
        if previous is None or current is None:
            return False
        original = current[0]()
        if (original is None or previous[0]() is not original
                or previous[1] is None or previous[1] != current[1]
                or current[1] != _signature(original)
                or current[2]() is not covariance
                or current[3] != _signature(covariance)
                or previous[5] != _signature(previous[4])
                or current[5] != _signature(current[4])
                or previous[4].shape != current[4].shape
                or previous[4].device != current[4].device):
            return False
        return torch.equal(previous[4], current[4])

    def lookup(self, covariance, weights):
        if self.result is None:
            _METRICS['reuse_misses'] += 1
            return None
        same_matrix = (self.matrix() is covariance
                       and self.matrix_signature == _signature(covariance))
        valid = ((same_matrix or self.same_transform(covariance))
                 and self.weight_signature == _signature(self.weights)
                 and self.result_signature == _signature(self.result)
                 and weights.shape == self.weights.shape
                 and weights.dtype == self.weights.dtype
                 and weights.device == self.weights.device)
        if valid:
            valid = torch.equal(weights.view(torch.int32),
                                self.weights.view(torch.int32))
        if not valid:
            _METRICS['reuse_input_changes'] += 1
            _METRICS['reuse_misses'] += 1
            self.clear()
            return None
        if covariance.is_cuda:
            from ._fit_structure_cache import guard_cuda_workspace
            guard_cuda_workspace(0, covariance.device,
                                 context='complete spectrum immutable scope hit')
        _METRICS['reuse_hits'] += 1
        self.current_binding = None
        return self.result

    def capture(self, covariance, weights, result):
        self.clear()
        signature = _signature(covariance)
        if signature is None or _signature(weights) is None or _signature(result) is None:
            _METRICS['reuse_unversioned_declines'] += 1
            self.current_binding = None
            return
        binding = self.current_binding
        if binding is not None and (binding[2]() is not covariance
                or binding[3] != signature or binding[0]() is None
                or binding[1] is None or binding[1] != _signature(binding[0]())):
            binding = None
        required = weights.untyped_storage().nbytes() + result.untyped_storage().nbytes()
        if binding is not None:
            required += binding[4].untyped_storage().nbytes()
        if self.limit is not None and required > self.limit:
            _METRICS['reuse_budget_declines'] += 1
            self.current_binding = None
            return
        if covariance.is_cuda:
            from ._fit_structure_cache import guard_cuda_workspace
            try:
                # These owners already exist. Check the extra retained lifetime
                # conservatively before an optional original-score reevaluation.
                guard_cuda_workspace(required, covariance.device,
                                     context='complete spectrum optional retained lifetime')
            except MemoryError:
                _METRICS['reuse_budget_declines'] += 1
                self.current_binding = None
                return
        self.matrix = weakref.ref(covariance)
        self.matrix_signature = signature
        self.weights = weights.detach()
        self.result = result.detach()
        self.weight_signature = _signature(self.weights)
        self.result_signature = _signature(self.result)
        self.binding = binding
        self.current_binding = None
        self.retained_bytes = required
        _METRICS['live_retained_bytes'] += required
        _METRICS['max_retained_spectrum_weight_bytes'] = max(
            _METRICS['max_retained_spectrum_weight_bytes'], required)
        _METRICS['reuse_captures'] += 1


@contextmanager
def same_covariance_spectrum_scope(*, enabled=True, cache_limit_bytes=None):
    """Reuse a complete spectrum only while the caller keeps C immutable.

    C must be the same Tensor object with unchanged storage/shape/mutation counter,
    or derive from that same original via the identical rare mask and unchanged
    FP32 subset/symmetric-mean operations. New D must be FP32-bit identical.
    NumPy/external writes bypassing Torch mutation
counters are outside this opt-in immutable-owner contract. No scope survives
a model evaluation, and no covariance Tensor or autograd graph is retained.
"""
    if type(enabled) is not bool or (cache_limit_bytes is not None and
            (type(cache_limit_bytes) is not int or cache_limit_bytes < 0)):
        raise ValueError('complete spectrum reuse needs boolean enabled and a nonnegative byte limit')
    previous = getattr(_LOCAL, 'reuse', None)
    state = _ReuseScope(cache_limit_bytes) if enabled else None
    _LOCAL.reuse = state
    _METRICS['reuse_scopes'] += int(enabled)
    try:
        yield
    finally:
        if state is not None:
            state.clear()
            state.current_binding = None
            _METRICS['closed_reuse_scopes'] += 1
        _LOCAL.reuse = previous
        state = previous = None


def current_reuse_scope():
    return getattr(_LOCAL, 'reuse', None)


def register_covariance_transform(original, transformed, selection):
    """Certify association_test's fixed rare-subset/symmetric FP32 transform."""
    state = current_reuse_scope()
    if state is not None:
        state.register_transform(original, transformed, selection)


def execution_metadata(*, reset=False):
    report = copy.deepcopy(_METRICS)
    report.update(spectrum_reuse='one immutable original C object/version, identical rare mask and fixed FP32 subset/symmetric transform, bit-identical D; no model/shape/proportional cache',
                  batch_relation_storage='bounded boolean tile plus J-by-J flags; no full M-by-J-by-J allocation',
                  spectrum_kind='complete H32; no eigenvalue truncation or approximate tail')
    if reset:
        if _METRICS['live_retained_bytes']:
            raise RuntimeError('cannot reset complete-spectrum metadata inside a live retained scope')
        for key in _METRICS:
            _METRICS[key] = 0
    return report
