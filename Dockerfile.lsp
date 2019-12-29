FROM python:3.8.0-buster

ARG mount_dir
WORKDIR /tmp
ADD requirements /tmp/requirements
WORKDIR /tmp/requirements
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt

RUN mkdir -p ${mount_dir}
WORKDIR ${mount_dir}
ADD setup.py ${mount_dir}/
ADD bowi ${mount_dir}/bowi
RUN pip install --editable .

VOLUME ${mount_dir}
