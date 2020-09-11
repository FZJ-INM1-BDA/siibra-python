FROM python:3.8-alpine

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

ADD . /brainscapes_client
WORKDIR /brainscapes_client

RUN pip install -r requirements.txt

# run unit tests as soon as available
