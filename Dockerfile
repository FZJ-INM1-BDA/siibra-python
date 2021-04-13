FROM python:3.8

#RUN apk update
#RUN apk add make automake gcc g++ subversion python3-dev
# Upgrade pip to latest version
RUN python -m pip install --upgrade pip

ADD . /siibra_client
WORKDIR /siibra_client

RUN pip install -r requirements.txt

RUN python -m unittest
