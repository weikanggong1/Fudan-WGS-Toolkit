"""Private immutable model stores and bounded, complete PheWAS CSV shards."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import weakref
import numpy as np
import torch
from .binary_null import BinaryNullModel
from .null_model import GaussianNullModel, KinshipSpectrum
from .tensor_validation import finite_check_scratch_bytes


_VERIFIED_FILES = {}


def verify_file(path, expected):
    path = Path(path)
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = str(path.resolve())
    prior = _VERIFIED_FILES.get(key)
    if prior is not None:
        if prior != (identity, expected):
            raise ValueError('immutable fitted-state identity changed')
        return
    if digest_file(path) != expected:
        raise ValueError('fitted-state checksum differs')
    _VERIFIED_FILES[key] = (identity, expected)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def _trait_index_bytes(phenotype_names, families):
    """Encode the derived index deterministically in source-column order."""
    names = list(phenotype_names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or not isinstance(families, Mapping)
            or set(families) != set(names)
            or any(families[name] not in ('gaussian', 'binomial') for name in names)):
        raise ValueError('trait index requires distinct names and their exact fitted families')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['trait_index', 'phenotype', 'family'])
    writer.writerows((index, name, families[name]) for index, name in enumerate(names))
    return stream.getvalue().encode('utf-8')


def trait_index_sha256(phenotype_names, families):
    """Return the expected checksum without creating an output directory."""
    return hashlib.sha256(_trait_index_bytes(phenotype_names, families)).hexdigest()


def write_trait_index(directory, phenotype_names, families, *, expected_sha256=None):
    """Verify or atomically repair the private, reproducible trait dictionary.

    This table is derived from already verified scientific inputs. Repair never
    changes model arrays or association blocks. An incorrect previous file or
    link is retained privately; an interrupted unique partial cannot prevent a
    later call in the same process from publishing the complete index.
    """
    payload = _trait_index_bytes(phenotype_names, families)
    expected = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and expected_sha256 != expected:
        raise ValueError('derived trait dictionary checksum differs from the scientific plan')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root/'trait_index.private.csv'
    if path.is_file() and not path.is_symlink() and digest_file(path) == expected:
        return dict(file=path.name, sha256=expected, bytes=len(payload), repaired=False)
    repaired = path.exists() or path.is_symlink()
    if repaired:
        diagnostic = root/'trait_index_invalid.private'/uuid.uuid4().hex
        diagnostic.mkdir(parents=True, mode=0o700)
        path.replace(diagnostic/path.name)
    partial = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.partial')
    with partial.open('xb') as stream:
        partial.chmod(0o600)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    if path.is_symlink() or digest_file(path) != expected:
        raise ValueError('published trait dictionary changed during verification')
    return dict(file=path.name, sha256=expected, bytes=len(payload), repaired=bool(repaired))


def save_model_store(model, destination, *, sample_rows, fit_metadata, spa_model=None):
    """Save fitted arrays as mmap-compatible NPY, retaining an FP64 SPA sidecar.

    Model data, sample IDs and source fields remain private. Normal association
    omits unused response/projection arrays, retaining the actual fitted X,
    Sigma_iX, precision, residual and covariance. Binary xw aliases Sigma_iX.T.
    The full original fitted SPA state is saved separately from native TF32.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('immutable fitted model store already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted notebook call can retry in the same process. Give each
    # unpublished attempt its own directory while retaining earlier partials.
    temp = destination.with_name(destination.name + f'.{os.getpid()}.{uuid.uuid4().hex}.partial')
    temp.mkdir()
    def save_state(current, directory, normal):
        directory.mkdir(exist_ok=True)
        family = current.family
        fields = ['x', 'scaled_residuals', 'coefficients', 'fixed_effect_covariance', 'precision_x']
        if family == 'gaussian':
            fields += ['theta', 'precision_theta', 'inverse_variance']
        else:
            fields += ['fitted_probability', 'precision']
            if not normal:
                fields += ['xw', 'projection_left', 'phenotype', 'working_phenotype']
        meta = dict(family=family, iterations=int(current.iterations), converged=bool(current.converged),
                    has_kinship=bool(current.has_kinship), matmul_mode=current.matmul_mode,
                    fit_method=getattr(current, 'fit_method', 'Gaussian_AI_REML'),
                    null_fit_source_sha256=getattr(current, 'null_fit_source_sha256', None),
                    use_spa=bool(current.use_spa), arrays={}, blocks=[])
        def array(name, value):
            value = np.asarray(value)
            np.save(directory/(name+'.npy'), value, allow_pickle=False)
            meta['arrays'][name] = dict(file=name+'.npy', shape=list(value.shape),
                                      dtype=value.dtype.str, sha256=digest_file(directory/(name+'.npy')))
        array('sample_ids', np.asarray(current.sample_ids, dtype=str))
        for name in fields:
            value = getattr(current, name, None)
            if value is None:
                continue
            if value.layout == torch.strided:
                array(name, value.detach().cpu().numpy())
            else:
                coo = value.to_sparse_coo().coalesce()
                array(name+'_indices', coo.indices().detach().cpu().numpy())
                array(name+'_values', coo.values().detach().cpu().numpy())
                meta[name+'_shape'] = list(coo.shape)
        if family == 'gaussian':
            array('eigenvalues', current.spectrum.eigenvalues.detach().cpu().numpy())
            for i, (rows, rotation) in enumerate(current.spectrum.blocks):
                array(f'block_{i}_rows', rows.detach().cpu().numpy())
                array(f'block_{i}_rotation', rotation.detach().cpu().numpy())
                meta['blocks'].append(i)
        atomic_json(directory/'state.private.json', meta)
        return meta
    normal = save_state(model, temp/'normal', True)
    if spa_model is not None:
        save_state(spa_model, temp/'spa', False)
    np.save(temp/'sample_rows.npy', np.asarray(sample_rows, dtype=np.int64), allow_pickle=False)
    atomic_json(temp/'model.private.json', dict(schema_version=1, fit=fit_metadata,
        n=model.n, family=model.family, normal_state='normal', spa_state='spa' if spa_model is not None else None,
        sample_rows_sha256=digest_file(temp/'sample_rows.npy'),
        state_sha256=digest_file(temp/'normal'/'state.private.json'),
        spa_state_sha256=digest_file(temp/'spa'/'state.private.json') if spa_model is not None else None))
    temp.replace(destination)
    return dict(n=model.n, family=model.family, path=str(destination),
                sha256=digest_file(destination/'model.private.json'))


def load_model_store(directory, *, device='cpu', spa=False, tensor_cache=None):
    """Load the supplied fitted state without refitting or modifying its arrays."""
    root = Path(directory)
    root_meta = json.loads((root/'model.private.json').read_text())
    directory = root/('spa' if spa else 'normal')
    expected = root_meta['spa_state_sha256' if spa else 'state_sha256']
    verify_file(directory/'state.private.json', expected)
    meta = json.loads((directory/'state.private.json').read_text())
    memo = {} if tensor_cache is None else tensor_cache
    def array(name):
        spec = meta['arrays'][name]
        filename = spec['file']
        if Path(filename).name != filename:
            raise ValueError('fitted-state path escaped its immutable directory')
        verify_file(directory/filename, spec['sha256'])
        value = np.load(directory/filename, mmap_mode='r', allow_pickle=False)
        if list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
            raise ValueError('fitted-state array geometry differs')
        return value
    def tensor(name):
        if name not in meta['arrays']:
            return None
        spec = meta['arrays'][name]
        key = (str(device), spec['sha256'], tuple(spec['shape']), spec['dtype'])
        # Only fixed design arrays are safely deduplicated across fitted states.
        cached = memo.get(key) if name == 'x' else None
        if cached is not None:
            return cached
        a = array(name)
        result = torch.tensor(np.asarray(a), device=device)
        if name == 'x':
            memo[key] = result
        return result
    def optional(name):
        if name in meta['arrays']:
            return tensor(name)
        if name+'_indices' in meta['arrays']:
            return torch.sparse_coo_tensor(tensor(name+'_indices'), tensor(name+'_values'),
                                          meta[name+'_shape'], device=device).coalesce()
        return None
    ids = np.asarray(array('sample_ids'))
    if meta['family'] == 'gaussian':
        kin = KinshipSpectrum(tensor('eigenvalues'), [(tensor(f'block_{i}_rows'),
              tensor(f'block_{i}_rotation')) for i in meta['blocks']])
        model = GaussianNullModel(ids, *(tensor(k) for k in ('x', 'scaled_residuals', 'coefficients',
              'theta', 'precision_theta', 'fixed_effect_covariance')), kin, tensor('inverse_variance'),
              tensor('precision_x'), meta['iterations'], meta['converged'],
              has_kinship=meta['has_kinship'], matmul_mode=meta['matmul_mode'])
    else:
        px = tensor('precision_x')
        model = BinaryNullModel(ids, tensor('x'), tensor('scaled_residuals'), tensor('fitted_probability'),
              tensor('xw') if spa else px.T, tensor('projection_left') if spa else None,
              tensor('fixed_effect_covariance'), precision=optional('precision'), precision_x=px,
              coefficients=tensor('coefficients'), phenotype=tensor('phenotype'),
              iterations=meta['iterations'], converged=meta['converged'], has_kinship=meta['has_kinship'],
              use_spa=meta['use_spa'], fit_method=meta['fit_method'], matmul_mode=meta['matmul_mode'],
              working_phenotype=tensor('working_phenotype'))
    model.null_fit_source_sha256 = meta.get('null_fit_source_sha256')
    return model


def load_spa_source(directory, *, device, tensor_cache):
    """Load verified minimal FP64 fitted arrays; derive large SPA projections on GPU."""
    root = Path(directory)
    root_meta = json.loads((root/'model.private.json').read_text())
    state = root/'spa'
    verify_file(state/'state.private.json', root_meta['spa_state_sha256'])
    meta = json.loads((state/'state.private.json').read_text())
    def array(name):
        spec = meta['arrays'][name]
        if Path(spec['file']).name != spec['file']:
            raise ValueError('SPA source array escaped its immutable directory')
        path = state/spec['file']
        verify_file(path, spec['sha256'])
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
            raise ValueError('SPA source array geometry differs')
        return value
    def tensor(name):
        spec = meta['arrays'][name]
        key = (str(device), spec['sha256'], tuple(spec['shape']), spec['dtype'])
        cached = tensor_cache.get(key) if name == 'x' else None
        if cached is not None:
            return cached
        result = torch.tensor(np.asarray(array(name)), device=device)
        if name == 'x':
            tensor_cache[key] = result
        return result
    if 'precision' in meta['arrays']:
        precision = tensor('precision')
    else:
        precision = torch.sparse_coo_tensor(tensor('precision_indices'), tensor('precision_values'),
                                             meta['precision_shape'], device=device).coalesce()
    return dict(sample_ids=np.asarray(array('sample_ids')), covariates=tensor('x'),
        residual=tensor('scaled_residuals'), fitted_probability=tensor('fitted_probability'),
        fixed_effect_covariance=tensor('fixed_effect_covariance'), precision=precision,
        null_fit_source_sha256=meta['null_fit_source_sha256'],
        verified_fit_source_sha256=meta['null_fit_source_sha256'], device=device,
        coefficients=tensor('coefficients') if 'coefficients' in meta['arrays'] else None,
        has_kinship=meta['has_kinship'], iterations=meta['iterations'], provenance=meta['fit_method'])


def _binary_source_peak_bytes(metadata):
    """Source upload plus its validation/projection or sparse-coalescing peak.

    Array shapes/dtypes come from the immutable source manifest. X deduplication
    may make this conservative; admission must precede loading any new arrays.
    The live reserve remains available for CUDA library-internal workspaces.
    """
    fields = {'x', 'scaled_residuals', 'fitted_probability', 'fixed_effect_covariance',
              'precision', 'precision_indices', 'precision_values', 'coefficients'}
    arrays = {name: spec for name, spec in metadata['arrays'].items() if name in fields}
    sizes = {name: int(np.prod(spec['shape']))*np.dtype(spec['dtype']).itemsize
             for name, spec in arrays.items()}
    n, q = arrays['x']['shape']
    # The compact source admission uses full isfinite, which can hold abs and
    # three boolean tensors. Later projection checks use the bounded helper.
    validation = max((int(np.prod(spec['shape']))*(np.dtype(spec['dtype']).itemsize+3)+2
                      for name, spec in arrays.items() if name != 'precision_indices'), default=0)
    sparse = sizes.get('precision_indices', 0)+sizes.get('precision_values', 0)
    # Coalescing can retain input/output indices and values plus sort buffers.
    # Constructor SX and Xcov checks are sequential, each uses one FP64 N*Q.
    projection_check = 8*n*q+finite_check_scratch_bytes((n, q), element_size=8)
    return sum(sizes.values())+max(2*sparse, validation, projection_check)


def _tensor_storage_bytes(tensor, *, dtype=None):
    if tensor is None:
        return 0
    element_size = tensor.element_size() if dtype is None else torch.finfo(dtype).bits//8
    if tensor.layout == torch.strided:
        return tensor.numel()*element_size
    # Fitted binary sources are coalesced COO; never estimate dense N-by-N.
    return tensor.indices().numel()*tensor.indices().element_size()+tensor.values().numel()*element_size


def _binary_derivation_peak_bytes(source, *, spa=False):
    """NEW bytes only: FP64 SX plus temporary normal casts or SPA Xcov.

    The immutable FP64 source is already allocated and counted by the allocator.
    FP32 sparse casts conservatively include their indices even if Torch aliases
    them. Normal xw aliases SX.T and SPA state aliases every other source array.
    """
    x, residual, probability, covariance, precision, coefficients, _ = source._tensors
    projection = 8*x.numel()
    if spa:
        return 2*projection
    normal = sum(_tensor_storage_bytes(value, dtype=torch.float32)
                 for value in (x, residual, probability, covariance, precision, coefficients))
    return projection+normal+4*x.numel()


class _WorkspaceReservation:
    """Future CUDA bytes, excluding tensors that are already allocated."""
    def __init__(self, repository, device, required_bytes, reserve_bytes, cache_release):
        self.repository, self.device = repository, torch.device(device)
        self.required_bytes, self.reserve_bytes = required_bytes, reserve_bytes
        self.cache_release, self.closed = cache_release, False

    def _check(self):
        if self.closed:
            raise RuntimeError('GPU workspace reservation is closed')

    def set_required(self, required_bytes, *, phase='workspace'):
        """Replace future bytes after a phase allocates or releases its buffers."""
        self._check()
        self.repository._validate_bytes(required_bytes, 'required_bytes')
        previous = self.required_bytes
        self.required_bytes = required_bytes
        try:
            self.ensure_free(0, phase=phase)
        except BaseException:
            self.required_bytes = previous
            raise

    def ensure_free(self, additional_bytes=0, *, phase='workspace'):
        """Admit the larger of this reservation and a stage's NEW allocation."""
        self._check()
        self.repository._validate_bytes(additional_bytes, 'additional_bytes')
        self.repository._admit(self.device, extra_bytes=additional_bytes, phase=phase)


class ModelRepository:
    """Bounded GPU fitted-state bank, with separate one-trait FP64 SPA state.

    The bank deduplicates exactly hashed X matrices. Allocation follows live
    device memory and leaves workspace for G/covariance; no fixed GiB ceiling
    applies. Entries held by nested acquire contexts cannot be evicted.
    """
    def __init__(self, entries, *, workspace_bytes=8*2**30,
                 binary_normal_cache='derived_fp64'):
        self._validate_bytes(workspace_bytes, 'workspace_bytes')
        if binary_normal_cache not in ('derived_fp64', 'stored_native'):
            raise ValueError('binary_normal_cache must be derived_fp64 or stored_native')
        self.entries = list(entries)
        self.workspace_bytes = workspace_bytes
        self.binary_normal_cache = binary_normal_cache
        self._closed = False
        self._workspace_reservations = []
        self._memory_policies = {}
        self._workspace_metrics = dict(workspace_admissions=0, workspace_evictions=0,
            workspace_cache_release_calls=0, workspace_required_bytes_highwater=0,
            workspace_additional_bytes_highwater=0, workspace_phase_counts={})
        # X remains deduplicated while any live model/source owns it. The memo
        # itself never pins a tensor after its last fitted state is retired.
        self.models, self.pinned = OrderedDict(), {}
        self.tensor_cache = weakref.WeakValueDictionary()
        self.spa_sources, self.spa_pinned = OrderedDict(), {}
        self.spa_source_loads = self.spa_source_hits = 0
        self.loads = self.hits = self.evictions = 0
        self._descriptors = []
        # Keep immutable sample axes in RAM rather than one open mmap per
        # phenotype. Exactly identical axes share storage; distinct complete
        # case sets retain their own order. Thousands of traits therefore do
        # not consume thousands of file descriptors before association starts.
        self._sample_axes = {}
        for i, entry in enumerate(self.entries):
            path = Path(entry['path'])
            verify_file(path/'model.private.json', entry['sha256'])
            model_meta = json.loads((path/'model.private.json').read_text())
            verify_file(path/'sample_rows.npy', model_meta['sample_rows_sha256'])
            axis_key = model_meta['sample_rows_sha256']
            if axis_key not in self._sample_axes:
                rows = np.load(path/'sample_rows.npy', allow_pickle=False)
                if rows.ndim != 1 or rows.dtype != np.dtype(np.int64):
                    raise ValueError('fitted sample rows must be a one-dimensional int64 axis')
                # A reversible readonly flag cannot prove immutable ownership.
                # Bytes-backed rows are safe aliases in descriptor grouping and
                # the shared broker, without retaining an open mmap per trait.
                self._sample_axes[axis_key] = np.frombuffer(rows.tobytes(), dtype=np.int64)
            rows = self._sample_axes[axis_key]
            if len(rows) != model_meta['n']:
                raise ValueError('fitted sample axis length differs from model metadata')
            self._descriptors.append(dict(trait_index=i, n=len(rows), family=entry['family'],
                ordered_rows=rows, projection_key=None, spa_available=(path/'spa').exists()))
    def descriptors(self):
        self._check_open()
        return self._descriptors

    def _check_open(self):
        if self._closed:
            raise RuntimeError('fitted-state bank is closed')

    def close(self):
        """Release only this bank's owned references, never immutable files.

        Active acquire/workspace contexts must finish first. External callers
        may still own a fitted tensor; allocator counters remain authoritative
        for such live storage after its bank reference has been retired.
        """
        if self._closed:
            return
        if (any(self.pinned.values()) or any(self.spa_pinned.values())
                or self._workspace_reservations or self._memory_policies):
            raise RuntimeError('cannot close fitted-state bank with active model/workspace contexts')
        self.models.clear()
        self.spa_sources.clear()
        self.tensor_cache.clear()
        self._sample_axes.clear()
        self._descriptors.clear()
        self.entries.clear()
        self.pinned.clear()
        self.spa_pinned.clear()
        self._closed = True
    @staticmethod
    def _validate_bytes(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(name+' must be a nonnegative integer')

    def _workspace_for(self, device):
        current = [r for r in self._workspace_reservations if r.device == device]
        # Separate live contexts represent distinct future allocations. The
        # ordinary bank headroom is retained when their combined need is small.
        required = max(self.workspace_bytes, sum(r.required_bytes for r in current))
        policy = self._memory_policies.get(device, (None, 0))
        reserve = max(policy[1], max((r.reserve_bytes for r in current), default=0))
        return required, reserve, current

    @contextmanager
    def memory_policy(self, device, *, limit_bytes=None, reserve_bytes=256*2**20):
        """Use the caller's process cap and live reserve during this run.

        None retains admission against current physical/live CUDA capacity.
        Nested contexts restore the previous policy on every exit.
        """
        self._check_open()
        if limit_bytes is not None:
            self._validate_bytes(limit_bytes, 'limit_bytes')
            if not limit_bytes:
                raise ValueError('limit_bytes must be positive or None')
        self._validate_bytes(reserve_bytes, 'reserve_bytes')
        device = torch.device(device)
        previous = self._memory_policies.get(device)
        self._memory_policies[device] = (limit_bytes, reserve_bytes)
        try:
            yield
        finally:
            if previous is None:
                self._memory_policies.pop(device)
            else:
                self._memory_policies[device] = previous

    def _admit(self, device, *, extra_bytes=0, fitted_bytes=0, phase='fitted_state', protect=()):
        """Evict owned, unpinned state until physical/live CUDA space admits NEW bytes.

        Unused allocator blocks may be reused; existing covariance, genotype and
        pinned models are already in the allocator counter and are never added
        to ``extra_bytes`` again. No external process or precision is changed.
        """
        device = torch.device(device)
        if device.type != 'cuda':
            return
        self._validate_bytes(extra_bytes, 'extra_bytes')
        self._validate_bytes(fitted_bytes, 'fitted_bytes')
        required, reserve, current = self._workspace_for(device)
        required = max(required, extra_bytes)+fitted_bytes
        metrics = self._workspace_metrics
        metrics['workspace_admissions'] += 1
        metrics['workspace_required_bytes_highwater'] = max(metrics['workspace_required_bytes_highwater'], required)
        metrics['workspace_additional_bytes_highwater'] = max(metrics['workspace_additional_bytes_highwater'], extra_bytes)
        phases = metrics['workspace_phase_counts']
        phases[phase] = phases.get(phase, 0)+1
        released = False
        while True:
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            unused = max(0, torch.cuda.memory_reserved(device)-allocated)
            limit = self._memory_policies.get(device, (None, 0))[0]
            cap = int(total) if limit is None else min(int(total), limit)
            available = max(0, min(int(free)+unused-reserve, cap-allocated))
            if required <= available:
                return
            if self._evict_one(device=device, protect=protect):
                metrics['workspace_evictions'] += 1
                torch.cuda.empty_cache()
                continue
            if not released:
                # CSR and fitted-state caches have independent owners. Their
                # callbacks only retire owned cached references, never call
                # this guard recursively or delete an active raw slab.
                for reservation in current:
                    if reservation.cache_release is not None:
                        reservation.cache_release()
                        metrics['workspace_cache_release_calls'] += 1
                released = True
                torch.cuda.empty_cache()
                continue
            raise MemoryError('live GPU memory cannot admit current fitted states plus '
                              f'{required} new workspace bytes ({phase}); pinned states are retained')

    @contextmanager
    def reserve_workspace(self, device, required_bytes, *, reserve_bytes=256*2**20, cache_release=None):
        """Temporarily replace static bank headroom with an actual job requirement.

        ``required_bytes`` counts only future allocations. Call ``set_required``
        after covariance allocation so its already resident bytes are excluded.
        Context exit restores outer reservations even after failed admission.
        """
        self._check_open()
        self._validate_bytes(required_bytes, 'required_bytes')
        self._validate_bytes(reserve_bytes, 'reserve_bytes')
        if cache_release is not None and not callable(cache_release):
            raise TypeError('cache_release must be callable or None')
        reservation = _WorkspaceReservation(self, device, required_bytes, reserve_bytes, cache_release)
        self._workspace_reservations.append(reservation)
        try:
            reservation.ensure_free(phase='job_workspace')
            yield reservation
        finally:
            self._workspace_reservations.remove(reservation)
            reservation.closed = True

    def _evict_one(self, *, device=None, protect=()):
        protected = set(protect)
        for key in list(self.models):
            if (key not in protected and not self.pinned.get(key, 0)
                    and (device is None or torch.device(key[1]) == device)):
                self.models.pop(key)
                self.evictions += 1
                return True
        for key in list(self.spa_sources):
            if (key not in protected and not self.spa_pinned.get(key, 0)
                    and (device is None or torch.device(key[1]) == device)):
                self.spa_sources.pop(key)
                self.evictions += 1
                return True
        return False
    @contextmanager
    def acquire(self, indices, device):
        from contextlib import ExitStack
        self._check_open()
        keys, models = [], []
        contexts = ExitStack()
        try:
            for i in indices:
                family = self._descriptors[i]['family']
                if family == 'binomial' and self.binary_normal_cache == 'derived_fp64':
                    source = contexts.enter_context(self._acquire_binary_source(int(i), device))
                    self._admit(device, fitted_bytes=_binary_derivation_peak_bytes(source),
                                phase='binary_normal_derivation')
                    models.append(source.normal())
                    continue
                key = (int(i), str(device))
                if key not in self.models:
                    if torch.device(device).type == 'cuda':
                        path = Path(self.entries[i]['path'])/'normal'
                        meta = json.loads((path/'state.private.json').read_text())
                        estimated = sum(int(np.prod(s['shape']))*np.dtype(s['dtype']).itemsize
                                        for name, s in meta['arrays'].items() if name != 'sample_ids')
                        if family == 'binomial':
                            # Native sparse precision loading may briefly hold
                            # input/output COO plus the coalescing sort buffers.
                            sparse = sum(int(np.prod(meta['arrays'][name]['shape']))*
                                np.dtype(meta['arrays'][name]['dtype']).itemsize
                                for name in ('precision_indices', 'precision_values')
                                if name in meta['arrays'])
                            estimated += 2*sparse
                        self._admit(device, fitted_bytes=estimated)
                    self.models[key] = load_model_store(self.entries[i]['path'], device=device,
                                                       tensor_cache=self.tensor_cache)
                    if family == 'binomial':
                        model = self.models[key]
                        if model.use_spa:
                            self.models.pop(key)
                            raise ValueError('stored native normal state must have SPA disabled')
                        model.normal_projection_reconstructed = False
                        model.normal_projection_method = 'authenticated_stored_'+model.matmul_mode
                        model.fitted_source_validation = 'authenticated_immutable_model_store'
                    self.loads += 1
                else:
                    self.hits += 1
                    # Protect this hit before admission; it must not be its
                    # own eviction victim or become a KeyError after a hit.
                    self._admit(device, protect=(key,))
                self.models.move_to_end(key)
                self.pinned[key] = self.pinned.get(key, 0)+1
                keys.append(key)
                models.append(self.models[key])
            yield models
        finally:
            contexts.close()
            for key in keys:
                self.pinned[key] -= 1
    @contextmanager
    def _acquire_binary_source(self, index, device):
        self._check_open()
        if not self._descriptors[index]['spa_available']:
            raise ValueError('binary SPA sidecar is missing')
        key = (int(index), str(device))
        if key not in self.spa_sources:
            meta = json.loads((Path(self.entries[index]['path'])/'spa'/'state.private.json').read_text())
            estimated = _binary_source_peak_bytes(meta)
            if torch.device(device).type == 'cuda':
                self._admit(device, fitted_bytes=estimated, phase='binary_fitted_source')
            from .binary_null import ImmutableFittedBinarySource
            self.spa_sources[key] = ImmutableFittedBinarySource(**load_spa_source(
                self.entries[index]['path'], device=device, tensor_cache=self.tensor_cache))
            self.spa_source_loads += 1
        else:
            self.spa_source_hits += 1
            self._admit(device, phase='binary_fitted_source_hit', protect=(key,))
        self.spa_sources.move_to_end(key)
        self.spa_pinned[key] = self.spa_pinned.get(key, 0)+1
        try:
            yield self.spa_sources[key]
        finally:
            self.spa_pinned[key] -= 1
    @contextmanager
    def acquire_spa(self, index, device):
        with self._acquire_binary_source(int(index), device) as source:
            self._admit(device, fitted_bytes=_binary_derivation_peak_bytes(source, spa=True),
                        phase='binary_spa_derivation')
            model = source.spa()
            try:
                yield model
            finally:
                del model
    def resident_storage_summary(self):
        """Endpoint owned tensor storage, deduplicating views and shared X.

        Walk once for diagnostics, including lazy model projection caches.
        This is not an allocator peak, total RSS or total card occupancy.
        """
        visited, storages = set(), {}
        def visit(value):
            if isinstance(value, torch.Tensor):
                if value.layout == torch.strided:
                    storage = value.untyped_storage()
                    key = (str(value.device), storage.data_ptr())
                    storages[key] = max(storages.get(key, 0), storage.nbytes())
                elif value.layout == torch.sparse_coo:
                    visit(value._indices())
                    visit(value._values())
                else:
                    # Fitted storage uses COO; account other sparse layouts
                    # without constructing a dense copy during diagnostics.
                    visit(value.values())
                    for name in ('crow_indices', 'col_indices', 'ccol_indices', 'row_indices'):
                        try:
                            visit(getattr(value, name)())
                        except RuntimeError:
                            pass
                return
            if id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)
            elif hasattr(value, '__dict__') and not isinstance(value, np.ndarray):
                visit(vars(value))
        visit(self.models)
        for source in self.spa_sources.values():
            visit(getattr(source, '_tensors', source))
        totals = {}
        for (device, _), size in storages.items():
            totals[device] = totals.get(device, 0)+size
        return dict(resident_storage_bytes_by_device=totals,
            unique_cuda_storage_bytes=sum(size for device, size in totals.items() if device.startswith('cuda')),
            resident_storage_scope='owned bank endpoint; unique physical tensor storage, including lazy projections')

    def summary(self):
        return dict(model_loads=self.loads, bank_hits=self.hits, evictions=self.evictions,
                    resident_models=len(self.models), workspace_bytes=self.workspace_bytes,
                    spa_source_loads=self.spa_source_loads, spa_source_hits=self.spa_source_hits,
                    resident_spa_sources=len(self.spa_sources), spa_projection_reconstructed_on_gpu=True,
                    binary_normal_from_same_fp64_source=self.binary_normal_cache == 'derived_fp64',
                    binary_normal_cache_policy=self.binary_normal_cache,
                    binary_normal_state_provenance=('derived_from_authenticated_fp64_source'
                        if self.binary_normal_cache == 'derived_fp64' else 'authenticated_stored_native'),
                    binary_spa_state_provenance='authenticated_original_fp64_source',
                    unique_sample_axes=len(self._sample_axes),
                    sample_axis_ram_bytes=sum(rows.nbytes for rows in self._sample_axes.values()),
                    sample_axis_open_mmaps=0,
                    sample_axis_immutable_backing='bytes',
                    live_deduplicated_design_tensors=len(self.tensor_cache),
                    active_workspace_reservations=len(self._workspace_reservations),
                    closed=self._closed,
                    **self.resident_storage_summary(),
                    **{key: dict(value) if isinstance(value, dict) else value
                       for key, value in self._workspace_metrics.items()})


def _rows(value):
    if value is None:
        return
    if isinstance(value, dict):
        if 'Chr' in value or 'CHR' in value or 'Gene name' in value or 'pvalue' in value:
            yield value
        else:
            for child in value.values():
                yield from _rows(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _rows(child)
    else:
        raise TypeError('association output must contain explicit record dictionaries')


class StreamingCSVWriter:
    """Complete CSV.gz shards and atomic receipts; no significance filtering."""
    def __init__(self, directory, *, worker_id='0', max_rows_per_shard=1_000_000,
                 max_open_streams=48, minimum_free_bytes=20*2**30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker_id, self.max_rows = str(worker_id), max_rows_per_shard
        self.max_open = max_open_streams
        self.minimum_free_bytes = minimum_free_bytes
        self.streams, self.sequence, self.receipts = OrderedDict(), {}, []
        self.row_counts, self.write_seconds = {}, 0.
    def _close(self, key):
        value = self.streams.pop(key)
        value['stream'].close()
        partial = value['path']
        final = partial.with_name(partial.name.removesuffix('.partial'))
        partial.replace(final)
        receipt = dict(kind=key[0], chromosome=key[1], rows=value['rows'],
            csv_file=str(final.relative_to(self.directory)), sha256=digest_file(final),
            compressed_bytes=final.stat().st_size, columns=value['columns'],
            first_job=value['first_job'], last_job=value['last_job'])
        atomic_json(final.with_name(final.name+'.receipt.json'), receipt)
        self.receipts.append(receipt)
    def write(self, kind, trait_index, chromosome, result, *, job_index=None):
        started = time.perf_counter()
        for source_row in _rows(result):
            row = dict(trait_index=int(trait_index), **source_row)
            columns = list(row)
            signature = hashlib.sha256(json.dumps(columns).encode()).hexdigest()[:12]
            key = (kind, str(chromosome), signature)
            if key in self.streams and (self.streams[key]['columns'] != columns or
                                       self.streams[key]['rows'] >= self.max_rows):
                self._close(key)
            if key not in self.streams:
                if shutil.disk_usage(self.directory).free < self.minimum_free_bytes:
                    raise OSError('CSV output filesystem is below its free-space reserve')
                while len(self.streams) >= self.max_open:
                    self._close(next(iter(self.streams)))
                seq = self.sequence.get(key, 0)
                self.sequence[key] = seq+1
                folder = self.directory/kind/f'chr{chromosome}'/signature
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/f'worker_{self.worker_id}.part_{seq:06d}.csv.gz.partial'
                if path.exists() or path.with_name(path.name.removesuffix('.partial')).exists():
                    raise FileExistsError('CSV writer refuses to overwrite a previous shard')
                stream = gzip.open(path, 'wt', encoding='utf-8', newline='', compresslevel=1)
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='raise')
                writer.writeheader()
                self.streams[key] = dict(path=path, stream=stream, writer=writer, columns=columns,
                                        rows=0, first_job=job_index, last_job=job_index)
            value = self.streams[key]
            value['writer'].writerow(row)
            value['rows'] += 1
            value['last_job'] = job_index
            self.streams.move_to_end(key)
            self.row_counts[kind] = self.row_counts.get(kind, 0)+1
        self.write_seconds += time.perf_counter()-started
    def finish(self):
        for key in list(self.streams):
            self._close(key)
        result = dict(schema_version=1, complete=True, worker_id=self.worker_id,
                      rows=self.row_counts, write_seconds=self.write_seconds,
                      files=len(self.receipts), receipts=self.receipts)
        atomic_json(self.directory/f'worker_{self.worker_id}.complete.private.json', result)
        return result
    def abort(self):
        """Close interrupted gzip streams, preserving partial shards as evidence."""
        partial = []
        for key in list(self.streams):
            value = self.streams.pop(key)
            value['stream'].close()
            partial.append(dict(path=str(value['path'].relative_to(self.directory)), rows=value['rows']))
        atomic_json(self.directory/f'worker_{self.worker_id}.interrupted.private.json',
                    dict(complete=False, closed_partial_streams=partial, completed_files=len(self.receipts)))
