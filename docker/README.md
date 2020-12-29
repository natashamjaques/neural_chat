# Running the project using Docker

To be able to run this project using docker, you will need [Docker](https://docs.docker.com/install/) and [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed in your machine

## Building the docker image

To create the docker image just run the `build.sh` script located in this folder, you will be prompted to download GloVe during the process, this won't be included in the docker image to avoid yielding a huge image, instead, it will be stored in the host machine and mounted as a volume in the container when running it.

## Before running the docker image

Before running the project inside docker, all extra files mentioned in the [base `README` file](../README.md) should be located in the same location mentioned there, they will be mounted as volumes.

You can either download them from your host system or execute the `run.sh` script with the same commands stated in the tutorial to download them using the docker image without having to potentially install any dependency in your host machine.

For example, you can run:

```bash
docker/run.sh python dataset_preprocess.py --dataset=cornell --shortcut
```

To download the Cornell dataset and place it in the `./datasets/cornell/` path in the host machine. The `datasets` folder will be mounted as a volume in the docker container when using the `run.sh` script.

## Running the project in the docker image
To run the image in interactive mode, just run
```bash
docker/run.sh bash
```

After that you will have a bash shell running inside the docker image with all the necessary files mounted as volumes, there you can interact with the project the same way you would do it in your machine.

To run actions in background, add -d as the first argument of the `run.sh` script.

## TL;DR

In order to have a docker image ready to run with the default configuration and the cornell dataset just follow this steps:

```bash
git clone git@github.com:natashamjaques/neural_chat.git
cd neural_chat
docker/build.sh
# Answer y to all questions
docker/run.sh python dataset_preprocess.py --dataset=cornell --shortcut
docker/run.sh bash
```

Now you are running a terminal inside the docker image with all the dependencies (including cuda and python requirements) installed and ready to start training the model without having to install anything besides docker and the Nvidia runtime in your host machine.