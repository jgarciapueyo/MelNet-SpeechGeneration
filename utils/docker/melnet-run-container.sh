#!/bin/bash

# Command to run the container using the bind mount functionality to make the directory of
# the host computer accessible from inside the container
docker run -it --rm --gpus all --mount src="$(pwd)",target=/app,type=bind jgarciapueyo/melnet