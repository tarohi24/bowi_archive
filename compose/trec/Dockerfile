FROM ubuntu:18.04

RUN apt-get update \
        && apt-get install -y git make gcc

RUN git clone https://github.com/usnistgov/trec_eval.git

WORKDIR /trec_eval

RUN make && make install
