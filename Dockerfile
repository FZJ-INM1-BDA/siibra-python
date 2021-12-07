# Inspired from https://www.docker.com/blog/containerized-python-development-part-1/

FROM jupyter/scipy-notebook:lab-3.2.3 as builder

USER root
RUN apt-get update
RUN apt-get install -y build-essential

RUN pip install -U pip

COPY . /siibra-python
WORKDIR /siibra-python
RUN pip install .

FROM jupyter/scipy-notebook:lab-3.2.3

COPY --from=builder /opt/conda /opt/conda

# HBP_AUTH_TOKEN
