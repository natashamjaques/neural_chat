FROM nvidia/cuda:10.0-runtime

WORKDIR /code

RUN apt-get -y update && apt-get install -y build-essential python3.6 python3.6-dev python3-pip wget

RUN ln -s /usr/bin/python3.6 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade ipython

RUN pip install --no-cache-dir torch===0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html

COPY requirements.txt /build/requirements.txt

RUN pip install --no-cache-dir -r /build/requirements.txt

COPY . /code/

RUN pip install -e .
