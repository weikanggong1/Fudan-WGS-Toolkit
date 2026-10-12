"""Frame-local variant labels shared by exact-cohort Single result tiles."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingleRecordMetadata:
    """Bind immutable selected indices and labels to one live reader.

    This owner belongs to a prepared Single block. It contains no model,
    dosage or device tensor and never survives that block's cleanup scope.
    Every consumer still checks the reader and the ordered variant indices.
    """
    reader: object
    indices: np.ndarray
    chromosome: object
    reference: object
    alternate: object

    @classmethod
    def prepare(cls, reader, indices):
        indices = np.asarray(indices, dtype=np.int64)
        bound = np.frombuffer(indices.tobytes(), dtype=np.int64)
        chromosome = reader.read_field('chromosome', indices)
        reference, alternate = reader.read_ref_alt(indices)
        if any(len(value) != len(bound) for value in (chromosome, reference, alternate)):
            raise ValueError('Single variant metadata must retain all selected columns')
        return cls(reader, bound, chromosome, reference, alternate)

    def values(self, reader, indices):
        if reader is not self.reader or not np.array_equal(indices, self.indices):
            raise ValueError('Single variant metadata reader or ordered columns changed')
        return self.chromosome, self.reference, self.alternate
