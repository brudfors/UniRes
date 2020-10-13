# Point to base image
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

MAINTAINER Mikael Brudfors <brudfors@gmail.com>

# Make bash the default shell
SHELL ["/bin/bash", "--login", "-c"]

# Install apps
RUN \
    apt-get update && \
    apt-get install -y wget git && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG username=docker
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN \
    adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# Set user
USER $USER

# Install miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O $HOME/miniconda.sh && \
    chmod +x $HOME/miniconda.sh && \
    $HOME/miniconda.sh -b -p $CONDA_DIR && \
    rm $HOME/miniconda.sh # Make non-activate conda commands available
# Make conda activate command available from /bin/bash --login shells
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda activate command available from /bin/bash --interative shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> $HOME/.profile 
RUN conda init bash

# Create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# Clone UniRes (private repo)
COPY --chown=$USER:$GID key $PROJECT_DIR
RUN \
    mkdir $HOME/.ssh && \
    ssh-keyscan -t rsa github.com >> $HOME/.ssh/known_hosts && \
    ssh-agent bash -c 'ssh-add $PROJECT_DIR/key; git clone git@github.com:brudfors/UniRes.git $PROJECT_DIR/unires'

# Clone nitorch (public repo)
RUN git clone https://github.com/balbasty/nitorch.git $PROJECT_DIR/nitorch

# Build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN \
    conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file $PROJECT_DIR/nitorch/conda/nitorch.yml --force && \
    conda clean --all --yes

# Install nitorch and unires
RUN \
    conda activate $ENV_PREFIX && \
    pip install $PROJECT_DIR/nitorch && \
    pip install $PROJECT_DIR/unires && \
    conda deactivate

# Copy entrypoint script
COPY --chown=$USER:$GID entrypoint.sh $PROJECT_DIR
RUN \
    chown $UID:$GID $PROJECT_DIR/entrypoint.sh && \
    chmod u+x $PROJECT_DIR/entrypoint.sh
    
# Commands to execute when running container
ENTRYPOINT ["/home/docker/app/entrypoint.sh"]
