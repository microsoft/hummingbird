# basic setup
FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev git
RUN pip3 -q install pip --upgrade

# Install HB
RUN pip3 install hummingbird-ml

# Install HB dev tools
RUN pip3 install flake8 coverage autopep8 jupyter pre-commit

# Jupyter
EXPOSE 8888

# Create a new system user
RUN useradd -ms /bin/bash jupyter

# Change to this new user
USER jupyter

# Pull repo to get notebooks
RUN cd ~/ &&  git clone https://github.com/microsoft/hummingbird.git

# Install precommit hooks
RUN cd ~/hummingbird && pre-commit install

# Set the container working directory to the user home folder
WORKDIR /home/jupyter/hummingbird/notebooks

# Start the jupyter notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=*"]
