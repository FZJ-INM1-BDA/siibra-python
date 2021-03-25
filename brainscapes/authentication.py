# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import environ
from . import logger


class Authentication(object):
    _instance = None
    _authentication_token = ''

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def get_token(self):
        if self._authentication_token == '':
            try:
                self._authentication_token = environ['HBP_AUTH_TOKEN']
            except KeyError:
                logger.warning('An authentication token must be set as an environment variable: HBP_AUTH_TOKEN')
        return self._authentication_token

    def set_token(self, token):
        self._authentication_token = token
