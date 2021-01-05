from distutils.core import setup
from setuptools import find_packages
import os
import sys

this = os.path.dirname(__file__)

packages = find_packages()
assert packages

# read version from the package file.
with (open(os.path.join(this, "hummingbird/__init__.py"), "r")) as f:
    line = [_ for _ in [_.strip("\r\n ") for _ in f.readlines()] if _.startswith("__version__")]
    if len(line) > 0:
        version_str = line[0].split("=")[1].strip('" ')

README = os.path.join(os.getcwd(), "README.md")
with open(README) as f:
    long_description = f.read()
    start_pos = long_description.find("## Introduction")
    if start_pos >= 0:
        long_description = long_description[start_pos:]

install_requires = [
    "numpy>=1.15,<=1.19.4",
    "onnxconverter-common>=1.6.0,<=1.7.0",
    "scipy<=1.5.4",
    "scikit-learn>=0.21.3,<=0.23.2",
    "torch>=1.4.*,<=1.7.1",
    "psutil",
    "dill",
]
onnx_requires = [
    "onnxruntime>=1.0.0",
    "onnxmltools>=1.6.0",
]
extra_requires = [
    # The need each for these depends on which libraries you plan to convert from
    "xgboost>=0.90",
    "lightgbm>=2.2,<3",
]
setup(
    name="hummingbird-ml",
    version=version_str,
    description="Convert trained traditional machine learning models into tensor computations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    author="Microsoft Corporation",
    author_email="hummingbird-dev@microsoft.com",
    url="https://github.com/microsoft/hummingbird",
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "tests": ["flake8", "pytest", "coverage", "pre-commit"],
        "sparkml": ["pyspark>=2.4.4"],
        "onnx": onnx_requires,
        "extra": extra_requires,
        "benchmark": onnx_requires + extra_requires + ["memory-profiler", "psutil"],
    },
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.5",
)
