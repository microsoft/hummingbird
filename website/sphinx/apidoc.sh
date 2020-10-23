#!/bin/bash
#
# A simple script to generate html docs from Python comments using sphinx.

# Be strict
set -eu

# Be verbose
set -x



HB_ROOT=$(readlink -f ".")


pythonCmd=python3

echo "Installing dependencies for generating Python API docs"

# Make sure we have up to date versions of the necessary packages (and their
# dependencies) rather than falling back to any system provided ones.
$pythonCmd -m pip install --upgrade sphinx sphinx_rtd_theme


echo "Generating Python API rst files"

sphinx-apidoc -o $HB_ROOT/website/sphinx/api -t $HB_ROOT/website/sphinx/_templates $HB_ROOT/hummingbird/ml  -d 1 -f -e
