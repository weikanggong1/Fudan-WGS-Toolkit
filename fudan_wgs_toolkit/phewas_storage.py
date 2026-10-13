"""Private immutable model stores and bounded, complete PheWAS CSV shards."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
import csv
import errno
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
from .npy_io import readonly_npy_mmap


_VERIFIED_FILES = {}
_FILE_VERIFICATION_METRICS = dict(calls=0, identity_hits=0, content_hashes=0,
                                  content_bytes=0, content_hash_seconds=0.0)


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verify_file(path, expected):
    path = Path(path)
    stat = path.stat()
    identity = _file_identity(stat)
    key = str(path.resolve())
    _FILE_VERIFICATION_METRICS['calls'] += 1
    prior = _VERIFIED_FILES.get(key)
    if prior is not None:
        if prior != (identity, expected):
            raise ValueError('immutable fitted-state identity changed')
        _FILE_VERIFICATION_METRICS['identity_hits'] += 1
        return identity
    start = time.perf_counter()
    actual = digest_file(path)
    _FILE_VERIFICATION_METRICS['content_hashes'] += 1
    _FILE_VERIFICATION_METRICS['content_bytes'] += stat.st_size
    _FILE_VERIFICATION_METRICS['content_hash_seconds'] += time.perf_counter()-start
    # Never associate bytes read while a file was replaced/modified with the
    # identity observed before the content scan.
    if _file_identity(path.stat()) != identity:
        raise ValueError('immutable fitted-state identity changed during verification')
    if actual != expected:
        raise ValueError('fitted-state checksum differs')
    _VERIFIED_FILES[key] = (identity, expected)
    return identity


def _sample_id_array(directory, spec, array_cache):
    """Share exact authenticated identity axes, without retaining model state.

    Every source path is verified even when another trait supplied the same
    NPY content. Bytes backing releases file descriptors and cannot be made
    writable; the weak cache does not pin axes after their final model owner.
    """
    filename = spec['file']
    if Path(filename).name != filename:
        raise ValueError('fitted-state path escaped its immutable directory')
    path = directory/filename
    verify_file(path, spec['sha256'])
    key = (spec['sha256'], tuple(spec['shape']), spec['dtype'])
    cached = None if array_cache is None else array_cache.get(key)
    if cached is not None:
        return cached
    value = readonly_npy_mmap(path)
    if list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
        raise ValueError('fitted-state array geometry differs')
    if value.dtype.kind not in 'US' or value.ndim != 1:
        raise ValueError('fitted sample identities must be a one-dimensional string axis')
    owned = np.frombuffer(value.tobytes(), dtype=value.dtype)
    verify_file(path, spec['sha256'])
    if array_cache is not None:
        array_cache[key] = owned
    return owned


class _SharedSparsePattern:
    """Weak sparse owners guard independently versioned aliases of one pattern.

    Separate COO tensors share index storage but have independent mutation
    counters. Check every still-live owner at a bank context boundary; checking
    only the canonical index wrapper would miss writes through another COO.
    """
    def __init__(self, indices):
        self.indices = indices
        self._index_stamp = self._stamp_indices(indices)
        self._owners = {}

    @staticmethod
    def _stamp_indices(indices):
        return (id(indices), indices.data_ptr(), indices._version,
                tuple(indices.shape), indices.dtype, indices.device)

    @staticmethod
    def _stamp_owner(owner):
        indices = owner.indices()
        return (owner._version, indices.data_ptr(), indices._version,
                tuple(indices.shape), tuple(owner.shape), owner.dtype,
                owner.device, owner.is_coalesced())

    def register(self, owner):
        token, reference = id(owner), weakref.ref(self)
        def retired(_):
            pattern = reference()
            if pattern is not None:
                pattern._owners.pop(token, None)
        self._owners[token] = (weakref.ref(owner, retired), self._stamp_owner(owner))
        owner._immutable_shared_pattern = self
        owner._immutable_shared_indices = self.indices

    def validate(self):
        if self._stamp_indices(self.indices) != self._index_stamp:
            raise ValueError('immutable shared sparse pattern was mutated')
        for reference, stamp in tuple(self._owners.values()):
            owner = reference()
            if owner is not None and self._stamp_owner(owner) != stamp:
                raise ValueError('immutable shared sparse owner was mutated')


def _guard_fitted_bank_context(method):
    """Check a shared pool once per outer acquire boundary, not per trait."""
    @wraps(method)
    @contextmanager
    def guarded(self, *args, **kwargs):
        if not self._fitted_context_depth:
            self._validate_shared_sparse_patterns()
        self._fitted_context_depth += 1
        try:
            with method(self, *args, **kwargs) as value:
                yield value
        finally:
            self._fitted_context_depth -= 1
            if not self._fitted_context_depth:
                self._validate_shared_sparse_patterns()
    return guarded


def _load_sparse_tensor(name, meta, directory, device, tensor_cache, tensor):
    """Reuse an exact sorted COO pattern; keep every trait's values separate.

    Only a pattern whose first coalesce preserves its input indices may be
    reused without a permutation/duplicate reduction. Other input patterns
    retain the original coalesce path. A COO owner keeps the weak memo's index
    wrapper alive, so shared storage follows fitted-bank eviction/closure.
    """
    spec = meta['arrays'][name+'_indices']
    filename = spec['file']
    if Path(filename).name != filename:
        raise ValueError('fitted-state path escaped its immutable directory')
    verify_file(directory/filename, spec['sha256'])
    shape = tuple(meta[name+'_shape'])
    key = ('coalesced_sparse_indices', str(device), spec['sha256'],
           tuple(spec['shape']), spec['dtype'], shape)
    pattern = tensor_cache.get(key)
    values = tensor(name+'_values')
    if pattern is not None:
        pattern.validate()
        result = torch.sparse_coo_tensor(pattern.indices, values, shape,
                                        device=device, is_coalesced=True)
    else:
        indices = tensor(name+'_indices')
        result = torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()
        if torch.equal(indices, result.indices()):
            pattern = _SharedSparsePattern(result.indices())
            tensor_cache[key] = pattern
    if pattern is not None:
        pattern.register(result)
    return result


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


def load_model_store(directory, *, device='cpu', spa=False, tensor_cache=None, array_cache=None):
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
        value = readonly_npy_mmap(directory/filename)
        verify_file(directory/filename, spec['sha256'])
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
            filename = spec['file']
            if Path(filename).name != filename:
                raise ValueError('fitted-state path escaped its immutable directory')
            verify_file(directory/filename, spec['sha256'])
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
            return _load_sparse_tensor(name, meta, directory, device, memo, tensor)
        return None
    ids = _sample_id_array(directory, meta['arrays']['sample_ids'], array_cache)
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


def load_spa_source(directory, *, device, tensor_cache, array_cache=None):
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
        value = readonly_npy_mmap(path)
        verify_file(path, spec['sha256'])
        if list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']:
            raise ValueError('SPA source array geometry differs')
        return value
    def tensor(name):
        spec = meta['arrays'][name]
        key = (str(device), spec['sha256'], tuple(spec['shape']), spec['dtype'])
        cached = tensor_cache.get(key) if name == 'x' else None
        if cached is not None:
            filename = spec['file']
            if Path(filename).name != filename:
                raise ValueError('SPA source array escaped its immutable directory')
            verify_file(state/filename, spec['sha256'])
            return cached
        result = torch.tensor(np.asarray(array(name)), device=device)
        if name == 'x':
            tensor_cache[key] = result
        return result
    if 'precision' in meta['arrays']:
        precision = tensor('precision')
    else:
        precision = _load_sparse_tensor('precision', meta, state, device, tensor_cache, tensor)
    return dict(sample_ids=_sample_id_array(state, meta['arrays']['sample_ids'], array_cache),
        covariates=tensor('x'),
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
                 binary_normal_cache='derived_fp64', cache_limit_bytes=None,
                 sample_axis_storage='mmap', sample_axis_mmap_limit=None,
                 trait_indices=None):
        self._validate_bytes(workspace_bytes, 'workspace_bytes')
        if cache_limit_bytes is not None:
            self._validate_bytes(cache_limit_bytes, 'cache_limit_bytes')
        if sample_axis_storage not in ('mmap', 'bytes'):
            raise ValueError('sample_axis_storage must be mmap or bytes')
        if sample_axis_mmap_limit is not None:
            self._validate_bytes(sample_axis_mmap_limit, 'sample_axis_mmap_limit')
        if binary_normal_cache not in ('derived_fp64', 'stored_native'):
            raise ValueError('binary_normal_cache must be derived_fp64 or stored_native')
        self.entries = list(entries)
        self._inventory_model_count = len(self.entries)
        self._explicit_trait_indices = trait_indices is not None
        selected = list(range(len(self.entries))) if trait_indices is None else list(trait_indices)
        if (any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                or i < 0 or i >= len(self.entries) for i in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('trait_indices must be distinct global model inventory indices')
        self._assigned_trait_indices = tuple(int(i) for i in selected)
        self.workspace_bytes = workspace_bytes
        self.binary_normal_cache = binary_normal_cache
        self.cache_limit_bytes = cache_limit_bytes
        self._cache_limits = {}
        self._cache_lru = OrderedDict()
        self._entry_storages = {}
        self._storage_owners = {}
        self._cache_bytes_by_device = {}
        self._cache_metrics = dict(cache_budget_evictions=0, cache_budget_active_overflows=0,
                                  cache_storage_bytes_highwater=0, file_descriptor_evictions=0,
                                  file_descriptor_retries=0)
        self._closed = False
        self._workspace_reservations = []
        self._fitted_context_depth = 0
        self._memory_policies = {}
        self._workspace_metrics = dict(workspace_admissions=0, workspace_evictions=0,
            workspace_cache_release_calls=0, workspace_required_bytes_highwater=0,
            workspace_additional_bytes_highwater=0, workspace_phase_counts={})
        # X remains deduplicated while any live model/source owns it. The memo
        # itself never pins a tensor after its last fitted state is retired.
        self.models, self.pinned = OrderedDict(), {}
        # Derived normals participate in the same physical-storage LRU as
        # stored normals and FP64 sources. Guards retain only scalar stamps
        # and a weak source reference, never otherwise unaccounted FP64 data.
        self._derived_normal_guards = {}
        self.binary_normal_derivations = self.binary_normal_cache_hits = 0
        self.tensor_cache = weakref.WeakValueDictionary()
        self.sample_id_cache = weakref.WeakValueDictionary()
        self._file_verification_baseline = dict(_FILE_VERIFICATION_METRICS)
        self.spa_sources, self.spa_pinned = OrderedDict(), {}
        self.spa_source_loads = self.spa_source_hits = 0
        self.loads = self.hits = self.evictions = 0
        self._descriptors = []
        self._descriptors_by_index = {}
        # Readonly mappings share the authenticated file pages across workers.
        # Leave descriptor headroom for readers/output: axes beyond that safe
        # budget fall back to immutable bytes without changing the process FD
        # limit. Exactly identical ordered axes always share one backing.
        self._sample_axes = {}
        self.sample_axis_storage = sample_axis_storage
        mmap_budget = self._sample_axis_mapping_budget()
        if sample_axis_mmap_limit is not None:
            mmap_budget = min(mmap_budget, sample_axis_mmap_limit)
        self._sample_axis_mmap_budget = mmap_budget
        mapped_axes = 0
        for i in self._assigned_trait_indices:
            entry = self.entries[i]
            path = Path(entry['path'])
            verify_file(path/'model.private.json', entry['sha256'])
            model_meta = json.loads((path/'model.private.json').read_text())
            verify_file(path/'sample_rows.npy', model_meta['sample_rows_sha256'])
            axis_key = model_meta['sample_rows_sha256']
            if axis_key not in self._sample_axes:
                mapped = sample_axis_storage == 'mmap' and mapped_axes < mmap_budget
                if mapped:
                    rows = readonly_npy_mmap(path/'sample_rows.npy')
                else:
                    with (path/'sample_rows.npy').open('rb', buffering=0) as stream:
                        rows = np.load(stream, allow_pickle=False)
                verify_file(path/'sample_rows.npy', model_meta['sample_rows_sha256'])
                if rows.ndim != 1 or rows.dtype != np.dtype(np.int64):
                    raise ValueError('fitted sample rows must be a one-dimensional int64 axis')
                self._sample_axes[axis_key] = (rows if mapped else
                                              np.frombuffer(rows.tobytes(), dtype=np.int64))
                mapped_axes += int(mapped)
            rows = self._sample_axes[axis_key]
            if len(rows) != model_meta['n']:
                raise ValueError('fitted sample axis length differs from model metadata')
            descriptor = dict(trait_index=i, n=len(rows), family=entry['family'],
                              sample_rows_sha256=axis_key,
                ordered_rows=rows, projection_key=None, spa_available=(path/'spa').exists())
            self._descriptors.append(descriptor)
            self._descriptors_by_index[i] = descriptor
    def descriptors(self):
        self._check_open()
        return self._descriptors

    def _descriptor(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError('trait index must be a global nonnegative integer')
        index = int(index)
        descriptor = self._descriptors_by_index.get(index)
        if descriptor is not None:
            return descriptor
        # Injected small contract banks historically replace _descriptors.
        # Production subsets always use the explicit global-index mapping.
        if not self._explicit_trait_indices and index < len(self._descriptors):
            descriptor = self._descriptors[index]
            if descriptor.get('trait_index', index) == index:
                return descriptor
        raise ValueError('trait index is not assigned to this model bank')

    def _check_open(self):
        if self._closed:
            raise RuntimeError('fitted-state bank is closed')

    def _validate_shared_sparse_patterns(self):
        for key, value in tuple(self.tensor_cache.items()):
            if key[0] == 'coalesced_sparse_indices':
                value.validate()

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
        self._derived_normal_guards.clear()
        self.spa_sources.clear()
        self.tensor_cache.clear()
        self.sample_id_cache.clear()
        self._sample_axes.clear()
        self._descriptors.clear()
        self._descriptors_by_index.clear()
        self.entries.clear()
        self.pinned.clear()
        self.spa_pinned.clear()
        self._cache_lru.clear()
        self._entry_storages.clear()
        self._storage_owners.clear()
        self._cache_bytes_by_device.clear()
        self._closed = True
    @staticmethod
    def _validate_bytes(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(name+' must be a nonnegative integer')

    @staticmethod
    def _file_descriptor_headroom():
        try:
            import resource
            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft == resource.RLIM_INFINITY:
                return None
            opened = len(os.listdir('/proc/self/fd'))
            return max(0, int(soft)-opened)
        except (ImportError, OSError, ValueError):
            return None

    @classmethod
    def _sample_axis_mapping_budget(cls):
        available = cls._file_descriptor_headroom()
        # Ordinary model sample IDs and portable metadata also retain maps.
        # Reserve half the available descriptors for those later consumers.
        return 4096 if available is None else max(0, (available-64)//2)

    def _admit_file_descriptors(self, device):
        while True:
            available = self._file_descriptor_headroom()
            if available is None or available >= 8:
                return
            if not self._evict_one(device=torch.device(device), namespace='normal'):
                raise OSError(errno.EMFILE, 'model loading cannot admit file descriptors; '
                              'active model and external mappings are retained')
            self._cache_metrics['file_descriptor_evictions'] += 1

    def _load_with_file_descriptor_admission(self, device, loader):
        while True:
            self._admit_file_descriptors(device)
            try:
                return loader()
            except OSError as error:
                if (error.errno != errno.EMFILE or
                        not self._evict_one(device=torch.device(device), namespace='normal')):
                    raise
                # A concurrent reader can consume descriptors after admission.
                # Each retry retires a distinct cold entry; it cannot spin once
                # all candidates are pinned or owned externally. Leave the
                # except scope before retrying so partial loads are released.
                self._cache_metrics['file_descriptor_retries'] += 1
                self._cache_metrics['file_descriptor_evictions'] += 1

    def _load_normal_model(self, index, device):
        verify_file(Path(self.entries[index]['path'])/'model.private.json',
                    self.entries[index]['sha256'])
        return self._load_with_file_descriptor_admission(device, lambda:
            load_model_store(self.entries[index]['path'], device=device,
                tensor_cache=self.tensor_cache, array_cache=self.sample_id_cache))

    @staticmethod
    def _owned_storages(value):
        visited, storages = set(), {}
        def visit(current):
            if isinstance(current, torch.Tensor):
                if current.layout == torch.strided:
                    backing = current.untyped_storage()
                    key = (str(current.device), backing.data_ptr())
                    storages[key] = max(storages.get(key, 0), backing.nbytes())
                elif current.layout == torch.sparse_coo:
                    visit(current._indices()); visit(current._values())
                else:
                    visit(current.values())
                    for name in ('crow_indices', 'col_indices', 'ccol_indices', 'row_indices'):
                        try:
                            visit(getattr(current, name)())
                        except (AttributeError, RuntimeError):
                            pass
                return
            if id(current) in visited:
                return
            visited.add(id(current))
            if isinstance(current, dict):
                for item in current.values():
                    visit(item)
            elif isinstance(current, (tuple, list)):
                for item in current:
                    visit(item)
            elif hasattr(current, '_tensors'):
                visit(current._tensors)
            elif hasattr(current, '__dict__') and not isinstance(current, np.ndarray):
                visit(vars(current))
        visit(value)
        return storages

    def _refresh_entry(self, namespace, key):
        token = (namespace, *key)
        bank = self.models if namespace == 'normal' else self.spa_sources
        new = self._owned_storages(bank[key])
        for storage in self._entry_storages.get(token, {}):
            owners = self._storage_owners[storage]
            before = max(owners.values())
            owners.pop(token, None)
            self._cache_bytes_by_device[storage[0]] -= before-(max(owners.values()) if owners else 0)
            if not owners:
                self._storage_owners.pop(storage)
        self._entry_storages[token] = new
        for storage, size in new.items():
            owners = self._storage_owners.setdefault(storage, {})
            before = max(owners.values(), default=0)
            owners[token] = size
            self._cache_bytes_by_device[storage[0]] = self._cache_bytes_by_device.get(storage[0], 0)+max(owners.values())-before
        self._cache_metrics['cache_storage_bytes_highwater'] = max(
            self._cache_metrics['cache_storage_bytes_highwater'],
            sum(self._cache_bytes_by_device.values()))

    def _touch_entry(self, namespace, key):
        token = (namespace, *key)
        self._refresh_entry(namespace, key)
        self._cache_lru[token] = None
        self._cache_lru.move_to_end(token)

    def _retire_entry(self, token):
        namespace, index, device = token
        bank = self.models if namespace == 'normal' else self.spa_sources
        bank.pop((index, device), None)
        if namespace == 'normal':
            self._derived_normal_guards.pop((index, device), None)
        self._cache_lru.pop(token, None)
        for storage in self._entry_storages.pop(token, {}):
            owners = self._storage_owners[storage]
            before = max(owners.values())
            owners.pop(token, None)
            self._cache_bytes_by_device[storage[0]] -= before-(max(owners.values()) if owners else 0)
            if not owners:
                self._storage_owners.pop(storage)
        self.evictions += 1

    def _synchronize_bank_index(self):
        # Ordinary acquisition updates the ledger directly. The length check
        # avoids walking the full thousand-trait bank on each workspace guard;
        # it also permits diagnostic/test callers to insert compatible entries.
        if len(self._cache_lru) == len(self.models)+len(self.spa_sources):
            return
        for namespace, bank in (('normal', self.models), ('spa', self.spa_sources)):
            for key in bank:
                if (namespace, *key) not in self._cache_lru:
                    self._touch_entry(namespace, key)
        for token in list(self._cache_lru):
            bank = self.models if token[0] == 'normal' else self.spa_sources
            if token[1:] not in bank:
                self._retire_entry(token)

    def _cache_bytes(self, device=None):
        self._synchronize_bank_index()
        return (sum(self._cache_bytes_by_device.values()) if device is None else
                self._cache_bytes_by_device.get(str(torch.device(device)), 0))

    def _trim_cache(self, device, *, protect=()):
        limit = self._cache_limits.get(torch.device(device), self.cache_limit_bytes)
        if limit is None:
            return
        while self._cache_bytes(device) > limit:
            if not self._evict_one(device=torch.device(device), protect=protect):
                self._cache_metrics['cache_budget_active_overflows'] += 1
                return
            self._cache_metrics['cache_budget_evictions'] += 1

    def _workspace_for(self, device):
        current = [r for r in self._workspace_reservations if r.device == device]
        # Separate live contexts represent distinct future allocations. The
        # ordinary bank headroom is retained when their combined need is small.
        required = max(self.workspace_bytes, sum(r.required_bytes for r in current))
        policy = self._memory_policies.get(device, (None, 0))
        reserve = max(policy[1], max((r.reserve_bytes for r in current), default=0))
        return required, reserve, current

    @contextmanager
    def memory_policy(self, device, *, limit_bytes=None, reserve_bytes=256*2**20,
                      cache_limit_bytes=None, cache_live_fraction=None):
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
        if cache_limit_bytes is not None:
            self._validate_bytes(cache_limit_bytes, 'cache_limit_bytes')
        if (cache_live_fraction is not None and (isinstance(cache_live_fraction, bool)
                or not isinstance(cache_live_fraction, (int, float))
                or not np.isfinite(cache_live_fraction) or not 0 < cache_live_fraction <= 1)):
            raise ValueError('cache_live_fraction must be in (0,1] or None')
        device = torch.device(device)
        previous = self._memory_policies.get(device)
        previous_cache_limit = self._cache_limits.get(device)
        self._memory_policies[device] = (limit_bytes, reserve_bytes)
        cache_budget = self.cache_limit_bytes if cache_limit_bytes is None else cache_limit_bytes
        if cache_live_fraction is not None and device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            unused = max(0, torch.cuda.memory_reserved(device)-allocated)
            cap = int(total) if limit_bytes is None else min(int(total), limit_bytes)
            # Own cached storage already consumes allocated memory. Restore
            # those bytes only to the cache sizing basis, never to admission.
            basis = max(0, min(int(free)+unused-reserve_bytes,
                               cap-allocated-reserve_bytes))+self._cache_bytes(device)
            live_budget = int(basis*cache_live_fraction)
            cache_budget = live_budget if cache_budget is None else min(cache_budget, live_budget)
        if cache_budget is not None:
            self._cache_limits[device] = cache_budget
        try:
            self._trim_cache(device)
            yield
        finally:
            if previous is None:
                self._memory_policies.pop(device)
            else:
                self._memory_policies[device] = previous
            if previous_cache_limit is None:
                self._cache_limits.pop(device, None)
            else:
                self._cache_limits[device] = previous_cache_limit

    def _admit(self, device, *, extra_bytes=0, fitted_bytes=0, phase='fitted_state', protect=()):
        """Evict owned, unpinned state until physical/live CUDA space admits NEW bytes.

        Unused allocator blocks may be reused; existing covariance, genotype and
        pinned models are already in the allocator counter and are never added
        to ``extra_bytes`` again. No external process or precision is changed.
        """
        device = torch.device(device)
        self._trim_cache(device, protect=protect)
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
            # Keep the reserve inside both the physical and process ceilings.
            # Subtracting it only from driver free space lets a smaller
            # process allocator cap be filled completely by planned tensors.
            available = max(0, min(int(free)+unused, cap-allocated)-reserve)
            if required <= available:
                return
            if self._evict_one(device=device, protect=protect):
                metrics['workspace_evictions'] += 1
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

    def _evict_one(self, *, device=None, protect=(), namespace=None):
        protected = set(protect)
        self._synchronize_bank_index()
        for token in list(self._cache_lru):
            entry_namespace, index, source_device = token
            key = (index, source_device)
            pins = self.pinned if entry_namespace == 'normal' else self.spa_pinned
            # Internal callers bind the namespace exactly. Older direct
            # callers using bare (index,device) conservatively protect both.
            if (token not in protected and key not in protected and not pins.get(key, 0)
                    and (namespace is None or namespace == entry_namespace)
                    and (device is None or torch.device(source_device) == torch.device(device))):
                self._retire_entry(token)
                return True
        return False
    @staticmethod
    def _normal_fitted_stamp(model):
        """Constant-size mutation/identity guard; lazy projection caches are separate."""
        from .binary_null import ImmutableFittedBinarySource
        fields = ('x', 'scaled_residuals', 'fitted_probability', 'xw',
                  'projection_left', 'fixed_effect_covariance', 'precision',
                  'precision_x', 'coefficients', 'phenotype', 'working_phenotype')
        tensors = tuple(ImmutableFittedBinarySource._stamp(getattr(model, name, None))
                        for name in fields)
        ids = model.sample_ids
        if not isinstance(ids, np.ndarray) or ids.flags.writeable:
            raise ValueError('derived binary normal sample identities must remain immutable')
        axis = (id(ids), ids.shape, ids.dtype.str, ids.strides,
                ids.__array_interface__['data'][0], ids.flags.writeable)
        metadata = tuple(getattr(model, name, None) for name in (
            'family', 'n_pheno', 'use_spa', 'matmul_mode', 'has_kinship',
            'null_fit_source_sha256', 'source_matmul_mode', 'iterations',
            'converged', 'fit_method'))
        return tensors, axis, metadata

    @staticmethod
    def _binary_source_stamp(source):
        current = tuple(source._stamp(value) for value in source._tensors)
        if current != source._stamps:
            raise ValueError('immutable binary fitted source was mutated')
        return current

    def _bind_derived_normal(self, key, model, source):
        binding = source.null_fit_source_sha256
        if (not isinstance(binding, str) or len(binding) != 64
                or any(ch not in '0123456789abcdef' for ch in binding)
                or getattr(model, 'null_fit_source_sha256', None) != binding):
            raise ValueError('derived binary normal differs from its authenticated fitted source')
        precision = getattr(model, 'precision', None)
        source_precision = source._tensors[4]
        pattern = getattr(source_precision, '_immutable_shared_pattern', None)
        if (pattern is not None and precision is not None
                and precision.layout == torch.sparse_coo
                and precision.indices().data_ptr() == pattern.indices.data_ptr()):
            pattern.register(precision)
        self._derived_normal_guards[key] = dict(
            fitted_source_sha256=binding, source=weakref.ref(source),
            source_stamp=self._binary_source_stamp(source),
            normal_stamp=self._normal_fitted_stamp(model))

    def _validate_derived_normal(self, key, model):
        guard = self._derived_normal_guards.get(key)
        if guard is None or self._normal_fitted_stamp(model) != guard['normal_stamp']:
            raise ValueError('cached derived binary normal fitted state was mutated')
        source = guard['source']()
        if source is not None:
            if (source.null_fit_source_sha256 != guard['fitted_source_sha256']
                    or self._binary_source_stamp(source) != guard['source_stamp']):
                raise ValueError('cached binary normal source identity changed')
        current = self.spa_sources.get(key)
        if current is not None and current is not source:
            self._validate_binary_source_binding(key, current)

    def _validate_binary_source_binding(self, key, source):
        guard = self._derived_normal_guards.get(key)
        if guard is not None:
            if source.null_fit_source_sha256 != guard['fitted_source_sha256']:
                raise ValueError('binary normal and SPA fitted-source bindings differ')
            self._binary_source_stamp(source)

    @_guard_fitted_bank_context
    @contextmanager
    def acquire(self, indices, device):
        self._check_open()
        keys, models = [], []
        try:
            for i in indices:
                family = self._descriptor(i)['family']
                key = (int(i), str(device))
                if family == 'binomial' and self.binary_normal_cache == 'derived_fp64':
                    if key not in self.models:
                        with self._acquire_binary_source(int(i), device) as source:
                            self._admit(device, fitted_bytes=_binary_derivation_peak_bytes(source),
                                        phase='binary_normal_derivation')
                            normal = source.normal()
                            self._bind_derived_normal(key, normal, source)
                            self.models[key] = normal
                            self.loads += 1
                            self.binary_normal_derivations += 1
                            self.pinned[key] = self.pinned.get(key, 0)+1
                            keys.append(key)
                            self._touch_entry('normal', key)
                            self._trim_cache(device)
                        # A normal tile owns its FP32 storage. Keeping its
                        # FP64 source pinned until the whole tile completes
                        # needlessly prevents the shared LRU from freeing it.
                        del source
                    else:
                        self._validate_derived_normal(key, self.models[key])
                        self.pinned[key] = self.pinned.get(key, 0)+1
                        keys.append(key)
                        self._admit(device, phase='binary_normal_cache_hit',
                                    protect=(('normal', *key),))
                        self.hits += 1
                        self.binary_normal_cache_hits += 1
                        self._touch_entry('normal', key)
                    self.models.move_to_end(key)
                    models.append(self.models[key])
                    continue
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
                    self.models[key] = self._load_normal_model(i, device)
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
                    self._admit(device, protect=(('normal', *key),))
                self.models.move_to_end(key)
                self.pinned[key] = self.pinned.get(key, 0)+1
                keys.append(key)
                self._touch_entry('normal', key)
                self._trim_cache(device)
                models.append(self.models[key])
            yield models
        finally:
            # Release every pin before bookkeeping so an exceptional nested
            # SPA or cache admission cannot leave unrelated normal pins live.
            for key in keys:
                self.pinned[key] -= 1
            for key in keys:
                self._refresh_entry('normal', key)
            self._trim_cache(device)
    @_guard_fitted_bank_context
    @contextmanager
    def _acquire_binary_source(self, index, device):
        self._check_open()
        if not self._descriptor(index)['spa_available']:
            raise ValueError('binary SPA sidecar is missing')
        key = (int(index), str(device))
        loaded = False
        if key not in self.spa_sources:
            meta = json.loads((Path(self.entries[index]['path'])/'spa'/'state.private.json').read_text())
            estimated = _binary_source_peak_bytes(meta)
            if torch.device(device).type == 'cuda':
                self._admit(device, fitted_bytes=estimated, phase='binary_fitted_source')
            verify_file(Path(self.entries[index]['path'])/'model.private.json',
                        self.entries[index]['sha256'])
            from .binary_null import ImmutableFittedBinarySource
            self.spa_sources[key] = self._load_with_file_descriptor_admission(device, lambda:
                ImmutableFittedBinarySource(**load_spa_source(self.entries[index]['path'],
                    device=device, tensor_cache=self.tensor_cache,
                    array_cache=self.sample_id_cache)))
            loaded = True
            self.spa_source_loads += 1
        else:
            self.spa_source_hits += 1
            self._admit(device, phase='binary_fitted_source_hit', protect=(('spa', *key),))
        try:
            self._validate_binary_source_binding(key, self.spa_sources[key])
        except BaseException:
            if loaded:
                # Validation happens before pin/admission into the ledger;
                # rejected reloads must not retain unaccounted FP64 storage.
                self.spa_sources.pop(key, None)
            raise
        self.spa_sources.move_to_end(key)
        self.spa_pinned[key] = self.spa_pinned.get(key, 0)+1
        try:
            self._touch_entry('spa', key)
            self._trim_cache(device)
            yield self.spa_sources[key]
        finally:
            self.spa_pinned[key] -= 1
            self._refresh_entry('spa', key)
            self._trim_cache(device)
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
        mapped = [rows for rows in self._sample_axes.values() if isinstance(rows, np.memmap)]
        anonymous = [rows for rows in self._sample_axes.values() if not isinstance(rows, np.memmap)]
        return dict(model_loads=self.loads, bank_hits=self.hits, evictions=self.evictions,
                    global_inventory_model_count=self._inventory_model_count,
                    assigned_model_count=len(self._assigned_trait_indices),
                    resident_models=len(self.models), workspace_bytes=self.workspace_bytes,
                    spa_source_loads=self.spa_source_loads, spa_source_hits=self.spa_source_hits,
                    resident_spa_sources=len(self.spa_sources), spa_projection_reconstructed_on_gpu=True,
                    binary_normal_from_same_fp64_source=self.binary_normal_cache == 'derived_fp64',
                    binary_normal_cache_policy=self.binary_normal_cache,
                    binary_normal_derivations=self.binary_normal_derivations,
                    binary_normal_cache_hits=self.binary_normal_cache_hits,
                    resident_derived_binary_normals=len(self._derived_normal_guards),
                    binary_normal_state_provenance=('derived_from_authenticated_fp64_source'
                        if self.binary_normal_cache == 'derived_fp64' else 'authenticated_stored_native'),
                    binary_spa_state_provenance='authenticated_original_fp64_source',
                    unique_sample_axes=len(self._sample_axes),
                    sample_axis_ram_bytes=sum(rows.nbytes for rows in anonymous),
                    sample_axis_mapped_logical_bytes=sum(rows.nbytes for rows in mapped),
                    sample_axis_open_mmaps=len(mapped),
                    sample_axis_immutable_backing=('readonly_mmap_with_immutable_bytes_fallback'
                        if self.sample_axis_storage == 'mmap' else 'bytes'),
                    sample_axis_mmap_budget=self._sample_axis_mmap_budget,
                    sample_axis_mapping_bytes_are_resident_memory=False,
                    fitted_cache_limit_bytes=self.cache_limit_bytes,
                    fitted_cache_limits_by_device={str(device): value for device, value in self._cache_limits.items()},
                    fitted_cache_unique_storage_bytes=self._cache_bytes(),
                    fitted_cache_eviction_policy='global_normal_and_spa_lru_with_active_pins',
                    live_deduplicated_design_tensors=sum(len(key) == 4 for key in self.tensor_cache),
                    live_deduplicated_sparse_index_tensors=sum(
                        key[0] == 'coalesced_sparse_indices' for key in self.tensor_cache),
                    live_deduplicated_sample_id_axes=len(self.sample_id_cache),
                    deduplicated_sample_id_bytes=sum(a.nbytes for a in self.sample_id_cache.values()),
                    file_verification_process_delta={key: value-self._file_verification_baseline[key]
                        for key, value in _FILE_VERIFICATION_METRICS.items()},
                    file_verification_metrics_scope='process cumulative delta since this bank opened',
                    active_workspace_reservations=len(self._workspace_reservations),
                    closed=self._closed,
                    **self.resident_storage_summary(),
                    **self._cache_metrics,
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
    _SCHEMA_CACHE_LIMIT = 128
    def __init__(self, directory, *, worker_id='0', max_rows_per_shard=1_000_000,
                 max_open_streams=48, minimum_free_bytes=20*2**30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker_id, self.max_rows = str(worker_id), max_rows_per_shard
        self.max_open = max_open_streams
        self.minimum_free_bytes = minimum_free_bytes
        self.streams, self.sequence, self.receipts = OrderedDict(), {}, []
        self.row_counts, self.write_seconds = {}, 0.
        # Only immutable column names/order and their signatures live here.
        # Never retain source rows or their array/tensor data owners.
        self._schema_cache = OrderedDict()
    def _schema(self, row):
        columns = tuple(row)
        signature = self._schema_cache.get(columns)
        if signature is None:
            signature = hashlib.sha256(json.dumps(list(columns)).encode()).hexdigest()[:12]
            self._schema_cache[columns] = signature
            if len(self._schema_cache) > self._SCHEMA_CACHE_LIMIT:
                self._schema_cache.popitem(last=False)
        self._schema_cache.move_to_end(columns)
        return columns, signature
    def _close(self, key):
        value = self.streams.pop(key)
        value['stream'].close()
        partial = value['path']
        final = partial.with_name(partial.name.removesuffix('.partial'))
        partial.replace(final)
        receipt = dict(kind=key[0], chromosome=key[1], rows=value['rows'],
            csv_file=str(final.relative_to(self.directory)), sha256=digest_file(final),
            compressed_bytes=final.stat().st_size, columns=list(value['columns']),
            first_job=value['first_job'], last_job=value['last_job'])
        atomic_json(final.with_name(final.name+'.receipt.json'), receipt)
        self.receipts.append(receipt)
    def write(self, kind, trait_index, chromosome, result, *, job_index=None):
        started = time.perf_counter()
        for source_row in _rows(result):
            row = dict(trait_index=int(trait_index), **source_row)
            columns, signature = self._schema(row)
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
