import os
import json
import requests

from string import Template

CLIENT_ID_env = 'KEYCLOAK_CLIENT_ID'
CLIENT_SECRET_env = 'KEYCLOAK_CLIENT_SECRET'
HBP_OIDC_ENDPOINT_env = 'HBP_KEYCLOAK_ENDPOINT'

HBP_TOKEN_env = 'HBP_AUTH_TOKEN'


def _check_envs():
    if "CI_PIPELINE" not in os.environ:
        if HBP_TOKEN_env not in os.environ:
            raise Exception("HBP_AUTH_TOKEN not set")
    else:
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
    if "CI_PIPELINE" in os.environ:
        result = requests.post(
            os.environ[HBP_OIDC_ENDPOINT_env],
            data = _build_request_object(),
            headers = {'content-type': 'application/x-www-form-urlencoded'}
        )
        token = None
        try:
            token = json.loads(result.content.decode("utf-8"))
        except json.JSONDecodeError as error:
            print("invalid json: %s" % error)
            raise Exception("Invalid response from OIDC")
        
        if 'error' in token:
            raise Exception(token['error_description'])
    else:
        token = {'access_token': os.environ[HBP_TOKEN_env]}

    return token

def main():
    print(get_token())


if __name__ == '__main__':
    main()
