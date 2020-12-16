#!/bin/bash

echo "Removing old dist directory"
rm -rf dist/*

echo "Installing and upgrading twine, setuptools and wheel"
python -m pip install --user --upgrade twine setuptools wheel

echo "Package the project"
python setup.py sdist bdist_wheel

# Decide wether to publish to prod or to test pypi
if [ "$1" = "prod" ]; then
  echo "Uploading to pypi"
  python -m twine upload dist/*
else
  echo "Uploading to test.pypi"
  python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi

