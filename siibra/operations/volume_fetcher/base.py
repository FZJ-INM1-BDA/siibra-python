from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from functools import reduce

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ...attributes.locations import BoundingBox

from ...operations.base import DataOp, Of
from ...operations.file_fetcher import RemoteLocalDataOp
from ...commons.logger import logger
from ...commons.conf import SIIBRA_MAX_FETCH_SIZE_GIB


class VolumeRetOp:

    @classmethod
    def transform_ops(
        cls, retrieval_ops: List[Dict], transform_ops: List[Dict], **kwargs
    ):
        assert issubclass(cls, DataOp)
        return [*retrieval_ops, cls.generate_specs(**kwargs)], transform_ops


class NgVolumeRetOp(VolumeRetOp):

    @classmethod
    def transform_ops(
        cls, retrieval_ops: List[Dict], transform_ops: List[Dict], **kwargs
    ):
        """
        From a prepopulated list of retrieval + transform ops, transform into a single
        ng fetch op.
        """

        from .neuroglancer_precomputed import ReadNeuroglancerPrecomputed

        assert issubclass(cls, DataOp)
        combined_ops = [*retrieval_ops, *transform_ops]

        assert len(combined_ops) > 0, f"Expected at least one operations"
        fetch_op, *rest_ops = combined_ops
        assert (
            fetch_op["type"] == RemoteLocalDataOp.type
        ), f"Expected the first op to be remote/local op"
        url = fetch_op["filename"]
        assert url, f"filename from remote local data op must be defined"

        starting_arg: Tuple[Union[None, Dict], List[Dict]] = (None, [])
        ng_fetch_op, transform_ops = reduce(
            cls.nifti_operations_reducer, rest_ops, starting_arg
        )
        retrieval_ops = []
        if ng_fetch_op is not None:
            retrieval_ops.append(ng_fetch_op)
        retrieval_ops.append(ReadNeuroglancerPrecomputed.generate_specs(url=url))
        return retrieval_ops, transform_ops

    @classmethod
    def nifti_operations_reducer(
        cls, acc: Tuple[Union[None, Dict], List[Dict]], curr: Dict
    ):
        """
        This method reduces a concatenated retrieval & transform ops, and return a a tuple consisting of:

        - fetch_kwargs = None | single retrieval_op conforming to NgPrecomputedFetchCfg
        - transform_ops = list of transformation_ops

        n.b. the starting value *must* be the same as accumulator (i.e. Tuple[Union[None, Dict], List[Dict]])

        At the moment, this reduce operation does the following:

        - if current item conforms to NgPrecomputedFetchCfg, set or merge with fetch_kwargs
        - if current item conforms to NiftiExtractVOI, convert to spec conforming NgPrecomputedFetchCfg, set or merge with fetch_kwargs
        - if current item conforms to ReadNiftiFromBytes, do nothing
        - in all other cases, append item to transform_ops

        Then return (fetch_kwargs, transform_ops)
        """
        from .nifti import (
            NiftiExtractVOI,
            ReadNiftiFromBytes,
        )
        from .neuroglancer_precomputed import NgPrecomputedFetchCfg
        from ...attributes.locations import BoundingBox

        ng_fetch_op, transform_ops = acc
        if curr.get("type") == NgPrecomputedFetchCfg.type:
            if ng_fetch_op is not None:
                logger.warning(
                    "multiple NgPrecomputedFetchCfg detected. Shadow overwrite"
                )
            return (
                NgPrecomputedFetchCfg.generate_specs(
                    fetch_config={
                        **(ng_fetch_op or {}).get("fetch_config", {}),
                        **curr.get("fetch_config", {}),
                    }
                ),
                transform_ops,
            )
        if curr.get("type") == ReadNiftiFromBytes.type:
            return ng_fetch_op, transform_ops
        if curr.get("type") != NiftiExtractVOI.type:
            return ng_fetch_op, [*transform_ops, curr]

        voi = curr.get("voi")

        assert isinstance(
            voi, BoundingBox
        ), f"Expected {curr} to be spec of NiftiExtractVOI, but value of 'voi' is not of type BoundingBox, but is {type(voi)} instead."

        if ng_fetch_op is None:
            return (
                NgPrecomputedFetchCfg.generate_specs(fetch_config={"bbox": voi}),
                transform_ops,
            )
        bbox = ng_fetch_op["fetch_config"].get("bbox")
        if bbox is None:
            return (
                NgPrecomputedFetchCfg.generate_specs(
                    fetch_config={"bbox": voi, **ng_fetch_op["fetch_config"]}
                ),
                acc,
            )
        assert isinstance(
            bbox, BoundingBox
        ), f"reducing nifti fetch op error. First op has type {curr[0]['type']}, but bbox is *not* of type BoundingBox, but is {type(bbox)}"
        bbox = bbox.intersect(voi)
        return (
            NgPrecomputedFetchCfg.generate_specs(
                fetch_config={**ng_fetch_op["fetch_config"], "bbox": bbox}
            ),
            transform_ops,
        )
