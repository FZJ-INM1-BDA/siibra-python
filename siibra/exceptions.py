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

"""Siibra specific exceptions"""


class SiibraException(Exception):
    pass


class ExcessiveArgumentException(ValueError, SiibraException):
    pass


class InsufficientArgumentException(ValueError, SiibraException):
    pass


class ConflictingArgumentException(ValueError, SiibraException):
    pass


class NonUniqueError(RuntimeError, SiibraException):
    pass


class NonUniqueIndexError(NonUniqueError, SiibraException):
    pass


class NoMapAvailableError(RuntimeError, SiibraException):
    pass


class NoVolumeFound(RuntimeError, SiibraException):
    pass


class WarmupRegException(SiibraException):
    pass


class AttrCompException(SiibraException):
    pass


class InvalidAttrCompException(AttrCompException, SiibraException):
    """
    Raised if try to signal the two Attributes do *not* match.

    (aka, does not match.)
    """

    pass


class UnregisteredAttrCompException(AttrCompException, SiibraException):
    """
    Raised if the Attributes have not been registered to match.

    (aka, we do not know how to match)
    """

    pass


class NotFoundException(SiibraException):
    pass


class ExternalApiException(RuntimeError, SiibraException):
    pass


class SpaceWarpingFailedError(ExternalApiException, SiibraException):
    pass


class SiibraTypeException(ValueError, SiibraException):
    pass
