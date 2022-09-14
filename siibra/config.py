import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.abspath(f"{ROOT_DIR}/VERSION"), "r") as fp:
    __version__ = fp.read()

USE_DEFAULT_PROJECT_TAG = "SIIBRA_CONFIG_GITLAB_PROJECT_TAG" not in os.environ

# Until openminds is fully supported, we get configurations of siibra concepts from gitlab.
GITLAB_PROJECT_TAG = os.getenv(
    "SIIBRA_CONFIG_GITLAB_PROJECT_TAG", "siibra-{}".format(__version__)
)

HBP_AUTH_TOKEN = os.getenv("HBP_AUTH_TOKEN")

KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

SIIBRA_CACHEDIR = os.getenv("SIIBRA_CACHEDIR")

SIIBRA_LOG_LEVEL = os.getenv("SIIBRA_LOG_LEVEL", "INFO")

SIIBRA_CONFIGURATION_SRC = os.getenv("SIIBRA_CONFIGURATION_SRC")
