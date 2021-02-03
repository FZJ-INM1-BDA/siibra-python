import os
import json
import requests

from string import Template

REFRESH_TOKEN_env = 'JUGEX_REFRESH_TOKEN'
CLIENT_ID_env = 'JUGEX_CLIENT_ID'
CLIENT_SECRET_env = 'JUGEX_CLIENT_SECRET'
HBP_OIDC_ENDPOINT_env = 'HBP_OIDC_ENDPOINT'


def _check_envs():
    if REFRESH_TOKEN_env not in os.environ:
        raise Exception("Refresh token not set")

    if CLIENT_ID_env not in os.environ:
        raise Exception("Client ID not set")

    if CLIENT_SECRET_env not in os.environ:
        raise Exception("Client secret not set")

    if HBP_OIDC_ENDPOINT_env not in os.environ:
        raise Exception("Endpoint URL not set")


def _build_request_object():
    request_template = Template('grant_type=refresh_token&refresh_token=${REFRESH_TOKEN}&client_id=${CLIENT_ID}&client_secret=${CLIENT_SECRET}')
    result = request_template.substitute(
                REFRESH_TOKEN = os.environ[REFRESH_TOKEN_env],
                CLIENT_ID = os.environ[CLIENT_ID_env],
                CLIENT_SECRET = os.environ[CLIENT_SECRET_env]
            )
    return result


def get_token():
    _check_envs()
    result = requests.post(
                os.environ[HBP_OIDC_ENDPOINT_env],
                data = _build_request_object(),
                headers = {'content-type': 'application/x-www-form-urlencoded'}
            )
    return result


def main():
    print(get_token())


if __name__ == '__main__':
    main()
