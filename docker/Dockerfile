
FROM nvcr.io/nvidia/pytorch:23.11-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y mc sudo git bash unattended-upgrades

RUN unattended-upgrade

# Install MMCV dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6 libgtk2.0-dev libgeos-dev cmake

# Install python packages

RUN pip install --upgrade setuptools

RUN git clone https://github.com/plantnet/malpolon.git
RUN cd ./malpolon &&\
    pip install -r requirements_python3.10.txt &&\
    pip install .

# Update Jupyterlab and install extentions
RUN pip install --upgrade jupyterlab
