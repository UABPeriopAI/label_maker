# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.224.2/containers/python-3-miniconda/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/devcontainers/python:0-${VARIANT}


# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends sqlite3 \
    && apt-get -y install --no-install-recommends graphviz \
    && apt-get -y install --no-install-recommends default-jre \
    && apt-get -y install --no-install-recommends cmake

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list &&\
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
#  COPY docs/dev_requirements.txt /tmp/pip-tmp/
#  RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/dev_requirements.txt \
#     && rm -rf /tmp/pip-tmp
    
USER vscode 


