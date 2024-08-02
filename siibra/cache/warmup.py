# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Union, NamedTuple, List
from enum import Enum
from functools import wraps
from filelock import FileLock as Lock
from concurrent.futures import ThreadPoolExecutor

from .cache import CACHE
from ..exceptions import WarmupRegException
from ..commons_new.logger import siibra_tqdm


class WarmupLevel(int, Enum):
    TEST = -1000
    INSTANCE = 1
    DATA = 5


class WarmupParam(NamedTuple):
    level: Union[int, WarmupLevel]
    fn: Callable
    is_factory: bool = False


class Warmup:

    _warmup_fns: List[WarmupParam] = []

    @staticmethod
    def fn_eql(wrapped_fn, original_fn):
        return wrapped_fn is original_fn or wrapped_fn.__wrapped__ is original_fn

    @classmethod
    def is_registered(cls, fn):
        return len([warmup_fn.fn
                    for warmup_fn in cls._warmup_fns
                    if cls.fn_eql(warmup_fn.fn, fn)]) > 0

    @classmethod
    def register_warmup_fn(cls, warmup_level: WarmupLevel = WarmupLevel.INSTANCE, *, is_factory=False):
        def outer(fn):
            if cls.is_registered(fn):
                raise WarmupRegException

            @wraps(fn)
            def inner(*args, **kwargs):
                return fn(*args, **kwargs)

            cls._warmup_fns.append(WarmupParam(warmup_level, inner, is_factory))
            return inner
        return outer

    @classmethod
    def deregister_warmup_fn(cls, original_fn):
        cls._warmup_fns = [
            warmup_fn for warmup_fn in cls._warmup_fns
            if not cls.fn_eql(warmup_fn.fn, original_fn)
        ]

    @classmethod
    def warmup(cls, warmup_level: WarmupLevel = WarmupLevel.INSTANCE, *, max_workers=4):
        all_fns = [warmup for warmup in cls._warmup_fns if warmup.level <= warmup_level]

        def call_fn(fn: WarmupParam):
            return_val = fn.fn()
            if not fn.is_factory:
                return
            for f in return_val:
                f()

        with Lock(CACHE.build_filename("lockfile", ".warmup")):
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for _ in siibra_tqdm(
                    ex.map(
                        call_fn,
                        all_fns
                    ),
                    desc="Warming cache",
                    total=len(all_fns),
                ):
                    ...

