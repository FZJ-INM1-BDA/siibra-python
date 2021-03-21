#!/bin/bash

echo "Removing old dist directory"
rm -rf dist/*

echo "Installing and upgrading twine, setuptools and wheel"
python3 -m pip install --user --upgrade twine setuptools wheel

echo "Package the project"
python3 setup.py sdist bdist_wheel

# Decide wether to publish to prod or to test pypi
if [ "$1" = "prod" ]; then
  echo "Uploading to pypi"
  python3 -m twine upload dist/*
else
  echo "Uploading to test.pypi"
  python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi

