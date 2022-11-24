import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.abspath(f"{ROOT_DIR}/VERSION"), "r") as fp:
    __version__ = fp.read()

HBP_AUTH_TOKEN = os.getenv("HBP_AUTH_TOKEN")

KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

SIIBRA_CACHEDIR = os.getenv("SIIBRA_CACHEDIR")
SIIBRA_LOG_LEVEL = os.getenv("SIIBRA_LOG_LEVEL", "INFO")
