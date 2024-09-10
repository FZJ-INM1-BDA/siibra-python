import pytest
from siibra.operations.base import DataOp
from siibra.operations.file_fetcher import RemoteLocalDataOp
from siibra.operations.volume_fetcher.base import NgVolumeRetOp
from siibra.operations.volume_fetcher.nifti import (
    NiftiExtractLabels,
    NiftiExtractVOI,
    ReadNiftiFromBytes,
)
from siibra.operations.volume_fetcher.neuroglancer_precomputed import (
    NgPrecomputedFetchCfg,
    ReadNeuroglancerPrecomputed,
)
from siibra.attributes.locations import BoundingBox


class Foo(DataOp, NgVolumeRetOp):
    type = None
    pass


remotelocal_dataop = RemoteLocalDataOp.generate_specs(filename="foo")
bbox = BoundingBox(minpoint=[0, 0, 0], maxpoint=[2, 2, 2], space_id="bar")
bbox_alt = BoundingBox(minpoint=[1, 1, 1], maxpoint=[5, 5, 5], space_id="bar")
bbox_result = BoundingBox(minpoint=[1, 1, 1], maxpoint=[2, 2, 2], space_id="bar")
bbox_invalid = BoundingBox(minpoint=[0, 0, 0], maxpoint=[1, 1, 1], space_id="foo")

args = [
    (
        [remotelocal_dataop],
        [],
        ([ReadNeuroglancerPrecomputed.generate_specs(url="foo")], []),
        None,
    ),
    (
        [
            remotelocal_dataop,
            ReadNiftiFromBytes.generate_specs(),
        ],
        [],
        ([ReadNeuroglancerPrecomputed.generate_specs(url="foo")], []),
        None,
    ),
    (
        [
            remotelocal_dataop,
            ReadNiftiFromBytes.generate_specs(),
        ],
        [
            NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
        ],
        (
            [ReadNeuroglancerPrecomputed.generate_specs(url="foo")],
            [NiftiExtractLabels.generate_specs(labels=[1, 2, 3])],
        ),
        None,
    ),
    (
        [
            remotelocal_dataop,
            ReadNiftiFromBytes.generate_specs(),
        ],
        [
            NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
            NiftiExtractVOI.generate_specs(voi=bbox),
        ],
        (
            [
                NgPrecomputedFetchCfg.generate_specs(fetch_config={"bbox": bbox}),
                ReadNeuroglancerPrecomputed.generate_specs(url="foo"),
            ],
            [
                NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
            ],
        ),
        None,
    ),
    (
        [
            remotelocal_dataop,
            ReadNiftiFromBytes.generate_specs(),
        ],
        [
            NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
            NiftiExtractVOI.generate_specs(voi=bbox),
            NiftiExtractVOI.generate_specs(voi=bbox_alt),
        ],
        (
            [
                NgPrecomputedFetchCfg.generate_specs(
                    fetch_config={"bbox": bbox_result}
                ),
                ReadNeuroglancerPrecomputed.generate_specs(url="foo"),
            ],
            [
                NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
            ],
        ),
        None,
    ),
    (
        [
            remotelocal_dataop,
            NgPrecomputedFetchCfg.generate_specs(fetch_config={"bbox": bbox}),
            NgPrecomputedFetchCfg.generate_specs(fetch_config={"bbox": bbox_alt}),
        ],
        [
            NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
        ],
        (
            [
                NgPrecomputedFetchCfg.generate_specs(fetch_config={"bbox": bbox_alt}),
                ReadNeuroglancerPrecomputed.generate_specs(url="foo"),
            ],
            [
                NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
            ],
        ),
        None,
    ),
    (
        [
            remotelocal_dataop,
            ReadNiftiFromBytes.generate_specs(),
        ],
        [
            NiftiExtractLabels.generate_specs(labels=[1, 2, 3]),
            NiftiExtractVOI.generate_specs(voi=bbox),
            NiftiExtractVOI.generate_specs(voi=bbox_invalid),
        ],
        None,
        Exception,
    ),
]


@pytest.mark.parametrize("retrieval_ops, transform_ops, expected, err", args)
def test_ngvolumeretop(retrieval_ops, transform_ops, expected, err):
    if err:
        with pytest.raises(err):
            Foo.transform_ops(retrieval_ops, transform_ops)
        return
    assert Foo.transform_ops(retrieval_ops, transform_ops) == expected
