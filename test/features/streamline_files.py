"""Writers of small synthetic TRX files for tests."""

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram

STREAMLINES = [
    (np.arange(n * 3, dtype="float32").reshape(n, 3) + 10 * i)
    for i, n in enumerate((3, 4, 2, 5, 3))
]
GROUPS = {"Left_Arcuate": [0, 1], "Left_Arcuate_anterior": [1, 2], "Right_Uncinate": [3, 4]}


def write_trx(path, streamlines=STREAMLINES, groups=GROUPS):
    """Write streamlines in RAS+ mm with groups (name -> indices) as an uncompressed TRX file."""
    from trx import trx_file_memmap as tmm

    header = nib.Nifti1Header()
    header.set_data_shape((100, 100, 100))
    header.set_sform(np.eye(4))
    trx = tmm.TrxFile.from_tractogram(Tractogram(streamlines, affine_to_rasmm=np.eye(4)), reference=header)
    trx.groups = {name: np.asarray(indices, dtype=np.uint32) for name, indices in groups.items()}
    tmm.save(trx, str(path))
    trx.close()
    return str(path)
