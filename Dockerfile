# basic setup
FROM python:3.6
RUN apt-get update && apt-get -y update
RUN apt-get install -y sudo git

# Setup user to not run as root
RUN adduser --disabled-password --gecos '' hb-dev
RUN adduser hb-dev sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER hb-dev

# Pull repo
RUN cd /home/hb-dev && git clone https://github.com/microsoft/hummingbird.git
WORKDIR /home/hb-dev/hummingbird

# Install HB  (Note: you may not need all of these packages and can remove some)
RUN sudo pip install -e .[docs,tests,extra,onnx]

# Install precommit hooks
RUN pre-commit install

# override default image starting point
CMD /bin/bash
ENTRYPOINT []