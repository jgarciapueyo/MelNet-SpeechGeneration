# Reference: https://pythonspeed.com/articles/activate-conda-dockerfile/
# Reference: https://github.com/Luke035/nvidia-anaconda-docker/blob/master/Dockerfile
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# install dependencies
RUN apt-get update && apt install -y \
	wget \
	bzip2 \
	make

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

# add conda to path
ENV PATH=/opt/conda/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# Create environment
COPY environment.yml .
RUN conda env create --name "melnet" --file=environment.yml

RUN mkdir -p /app
WORKDIR /app

# Activate the environment every time when logging inside the container
RUN	echo "source activate melnet" >> /root/.bashrc

