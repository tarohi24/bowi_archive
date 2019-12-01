FROM buildpack-deps:18.04
SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
        apt-get install -y gcc build-essential libomp-dev libopenblas-dev cmake pkg-config gfortran git wget curl zlib1g-dev libssl-dev

ARG homedir
ENV HOME=${homedir}

# Create user
RUN useradd -u 1001 -m jovyan
USER jovyan
# Declare environment variables  (Locate them *after* user creation)
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH=${PYENV_ROOT}/bin:${PATH}
ENV PYTHON_VERSION=3.8.0


# Install pyenv
RUN git clone --recursive --shallow-submodules \
        https://github.com/pyenv/pyenv.git \
        $PYENV_ROOT
RUN pyenv install ${PYTHON_VERSION} && \
        pyenv global ${PYTHON_VERSION}

# Instal dependencies
ARG CACHEBUST=1
WORKDIR ${HOME}
ADD scripts ${HOME}/scripts
ADD requirements ${HOME}/requirements
RUN bash scripts/pip_install.bash

RUN echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
