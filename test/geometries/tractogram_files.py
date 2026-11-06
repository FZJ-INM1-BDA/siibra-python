"""Writers of small synthetic tractogram files for tests."""

import numpy as np
import nibabel as nib
from nibabel.streamlines import Field, Tractogram, TrkFile

N_POINTS = (3, 4, 2, 5, 3)  # vertices per streamline


def make_streamlines(seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random((n, 3)) * 50).astype("float32") for n in N_POINTS]


def write_trx(path, streamlines, groups):
    """Write streamlines in RAS+ mm with the given groups (name -> indices) as uncompressed TRX."""
    from trx import trx_file_memmap as tmm

    header = nib.Nifti1Header()
    header.set_data_shape((100, 100, 100))
    header.set_sform(np.eye(4))
    trx = tmm.TrxFile.from_tractogram(Tractogram(streamlines, affine_to_rasmm=np.eye(4)), reference=header)
    trx.groups = {name: np.asarray(indices, dtype=np.uint32) for name, indices in groups.items()}
    tmm.save(trx, str(path))
    trx.close()
    return str(path)


def write_trk(path, streamlines):
    header = {
        Field.VOXEL_TO_RASMM: np.eye(4),
        Field.DIMENSIONS: np.array([100, 100, 100], dtype="int16"),
        Field.VOXEL_SIZES: np.ones(3, dtype="float32"),
    }
    TrkFile(Tractogram(streamlines, affine_to_rasmm=np.eye(4)), header=header).save(str(path))
    return str(path)
