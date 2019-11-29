FROM python:3.8-buster

WORKDIR /workplace
RUN apt-get update && \
        apt-get install -y gcc build-essential libomp-dev libopenblas-dev cmake pkg-config gfortran

RUN pip install --upgrade pip

# install sentencepiece
WORKDIR /tmp
RUN git clone https://github.com/google/sentencepiece.git
RUN mkdir ./sentencepiece/build
WORKDIR ./sentencepiece/build
RUN cmake .. && make -j $(nproc) && make install
WORKDIR ../python
RUN python setup.py build && python setup.py install

ARG CACHEBUST=1
WORKDIR /workplace
RUN pip install numpy scipy Cython
ADD requirements_mine.txt /workplace/
RUN pip install -r requirements_mine.txt
ADD requirements_dev.txt /workplace/
RUN pip install -r requirements_dev.txt
ADD requirements.txt /workplace/
RUN pip install -r requirements.txt
# NLP Kits
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
