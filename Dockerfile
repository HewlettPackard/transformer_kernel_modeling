FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

LABEL maintainer="Pedro Bruel <bruel@hpe.com>"

ARG PROJECT_NAME
ARG USER_NAME

ENV DEBIAN_FRONTEND=noninteractive
ENV R_LIBS_USER=/usr/local/lib/R/site-library
ENV MAKEFLAGS="-j16"

RUN apt-get update --yes \
    && apt-get upgrade --yes \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    texlive-latex-extra \
    software-properties-common \
    dirmngr \
    wget \
    ca-certificates\
    sudo \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
    | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc

RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/"

RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY requirements.R .

RUN Rscript requirements.R

USER ${USER_NAME}

WORKDIR /home/${USER_NAME}/${PROJECT_NAME}
