#!/usr/bin/env bash

DIR=$(dirname "$(realpath $0)")
PROJECT_ROOT="$(realpath $DIR/..)"
DOCKER_IMAGE="neuralchat"

if [[ "$(docker images -q $DOCKER_IMAGE:latest 2> /dev/null)" == "" ]]; then
  echo "Docker image does not exist, please build it using the build.sh script"
  exit 1
fi

# Docker volumes for extra data that is too big to be included in the docker image
MODELS_VOLUME="$PROJECT_ROOT/datasets/:/code/datasets"
GLOVE_VOLUME="$PROJECT_ROOT/inferSent/dataset/GloVe/:/code/inferSent/dataset/GloVe/"
CHECKPOINTS_VOLUME="$PROJECT_ROOT/model_checkpoints/:/code/model_checkpoints"
INFERSENT_PICKLE_VOLUME="$PROJECT_ROOT/inferSent/encoder/infersent1.pickle:/code/inferSent/encoder/infersent1.pickle"

if [ $# = 0 ]; then
  echo "Usage: $0 [-d] <command>"
  exit 1
fi

if [ $1 = "-d" ]; then
  DETACH="True"
  shift
else
  IT="True"
fi

docker run --gpus all ${DETACH:+-d} ${IT:+-it} -v $MODELS_VOLUME -v $GLOVE_VOLUME -v $CHECKPOINTS_VOLUME -v $INFERSENT_PICKLE_VOLUME $DOCKER_IMAGE:latest $@
