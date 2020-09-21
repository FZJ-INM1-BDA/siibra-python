from os import environ


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
                print('An authentication token must be set as an environment variable: HBP_AUTH_TOKEN')
        return self._authentication_token

    def set_token(self, token):
        self._authentication_token = token
