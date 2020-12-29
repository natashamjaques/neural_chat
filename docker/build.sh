#!/usr/bin/env bash
set -ue

DIR=$(dirname "$(realpath $0)")
PROJECT_ROOT="$(realpath $DIR/..)"
DOCKER_IMAGE="neuralchat"

function get_glove {
    GLOVE_DIR=$PROJECT_ROOT/inferSent/dataset/GloVe
    mkdir -p $GLOVE_DIR
    if ! [ -f $GLOVE_DIR/glove.840B.300d.txt ]; then
        curl -Lo $GLOVE_DIR/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
        unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR
        rm -f $GLOVE_DIR/glove.840B.300d.zip
    else
        echo "GloVe file is already present in $GLOVE_DIR"
    fi

    if ! [ -f $PROJECT_ROOT/inferSent/encoder/infersent1.pickle ]; then
        curl -Lo $PROJECT_ROOT/inferSent/encoder/infersent1.pickle https://affect.media.mit.edu/neural_chat/inferSent/encoder/infersent1.pickle
    else
        echo "$PROJECT_ROOT/inferSent/encoder/infersent1.pickle already exists"
    fi
}

function get_torchmoji {
    docker run -it -v $PROJECT_ROOT/torchMoji/model:/code/torchMoji/model $DOCKER_IMAGE:latest python torchMoji/scripts/download_weights.py
}

echo -n "Do you want to download GloVe[y/N]? "
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    get_glove
else
    echo "GLoVe will not be downloaded, be sure to put it in the correct location before running the docker image"
fi

cd $PROJECT_ROOT && docker build -f $PROJECT_ROOT/docker/Dockerfile -t $DOCKER_IMAGE:latest . && cd -

echo -n "Do you want to download torchMoji weights and include them in the image [y/N]? "
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then
    get_torchmoji
    cd $PROJECT_ROOT && docker build -f $PROJECT_ROOT/docker/Dockerfile -t $DOCKER_IMAGE:latest . && cd -
else
    echo "torchMoji will not be downloaded, be sure to put it in the correct location before running the docker image and add the corresponding volume to the run file"
fi


