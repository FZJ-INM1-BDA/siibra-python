from typing import Dict, Type
from ...operations import DataOp
from ...commons.logger import logger


class VolumeRetOp:
    @classmethod
    def get_pre_retrieval_ops(cls, **kwargs):
        assert issubclass(cls, DataOp)
        return []

    @classmethod
    def get_post_retrieval_ops(cls, **kwargs):
        assert issubclass(cls, DataOp)
        return [cls.generate_specs(**kwargs)]


class NgVolumeRetOp(VolumeRetOp):
    @classmethod
    def get_pre_retrieval_ops(cls, **kwargs):
        assert issubclass(cls, DataOp)
        # append a noop, so that default fetching does not occur
        return [DataOp.generate_specs()]

    @classmethod
    def get_post_retrieval_ops(cls, *, url, **kwargs):
        assert issubclass(cls, DataOp)
        return [cls.generate_specs(url=url, **kwargs)]
