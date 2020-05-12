# basic setup
FROM oryxprod/python-3.6
RUN apt-get update && apt-get -y update
RUN pip -q install pip --upgrade

# Install dev dependencies
RUN pip -q install flake8 coverage autopep8 pre-commit pylint
