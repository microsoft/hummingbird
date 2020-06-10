#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.6

#
# Update the OS and maybe install packages
#
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && apt-get upgrade -y  \
   && apt-get -y install --no-install-recommends build-essential \
   && apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

#
# Install extras for development
#
RUN pip3 --disable-pip-version-check --no-cache-dir install hummingbird-ml[docs,tests,extra] autopep8
