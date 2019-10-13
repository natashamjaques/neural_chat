# Running the project using Docker

To be able to run this project using docker, you will need [Docker](https://docs.docker.com/install/) and [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed in your machine

## Building the docker image

To create the docker image just run the `build.sh` script located in this folder, you will be prompted to download GloVe during the process, this won't be included in the docker image to avoid yielding a huge image, instead, it will be mounted as a volume in the container when running it.

## Before running the docker image

Before running the project inside docker, all extra files mentioned in the base `README` file should be located in the same location mentioned in the `README`, they will be mounted as volumes.

You can either download them from your host system or execute the `run.sh` script with the same commands to download them using the docker image without having to install the possible dependencies.

For example, you can run:

```bash
docker/run.sh python dataset_preprocess.py --dataset=cornell --shortcut
```

To download the Cornell dataset and place it in the `./datasets/cornell/` path

## Running the project in the docker image
To run the image in interactive mode, just run
```bash
docker/run.sh bash
```

After that you will have a bash shell running inside the docker image with all the necessary files mounted as volumes, there you can interact with the project the same way you would do it in your machine.

To run actions in background, add -d as the first argument of the `run.sh` script.
