FROM python:3.8

# Command to run a build and all tests in docker.
# A test coverage report will be created at the end of the build.
# docker build --progress=plain -t siibra-python -f Dockerfile --build-arg JUGEX_CLIENT_ID={...} --build-arg JUGEX_CLIENT_SECRET={...} --build-arg JUGEX_REFRESH_TOKEN={...} .

ARG JUGEX_CLIENT_ID
ARG JUGEX_CLIENT_SECRET
ARG JUGEX_REFRESH_TOKEN
ARG HBP_OIDC_ENDPOINT='https://iam.ebrains.eu/auth/realms/hbp/protocol/openid-connect/token'
ARG CI_PIPELINE=1

RUN python -m pip install --upgrade pip

ADD . /siibra-python
WORKDIR /siibra-python

RUN pip install -r requirements.txt
RUN pip install pytest pytest-cov coverage

RUN python -m unittest
RUN coverage run -m unittest
RUN coverage report
