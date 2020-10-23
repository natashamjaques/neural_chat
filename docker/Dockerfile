FROM nvidia/cuda:10.0-cudnn7-runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /code

#RUN apt-get -y update && \
#    apt-get install -y --no-install-recommends software-properties-common && \
#    add-apt-repository -y ppa:deadsnakes/ppa && \
#    apt-get --purge remove -y software-properties-common && \
#    rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends wget python3.6 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.6 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --no-cache-dir --upgrade ipython

RUN pip install --no-cache-dir torch===0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

COPY requirements.txt /build/requirements.txt

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends build-essential python3.6-dev && \
    python -m pip install --no-cache-dir -r /build/requirements.txt && \
    apt-get -y --purge remove build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . /code/

RUN python -m pip install -e .

EXPOSE 6009
