.PHONY: clean-pyc data

# found in https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

# starts the program to download the librispeech dataset to MelNet-SpeechGeneration/data/
data-librispeech:
	python src/data/librispeech.py -r data/

# starts training on small MelNet model to test locally if training runs correctly
dummy-train:
	python src/train.py -p models/params/dummymodel_podcast.yml

# starts training on MelNet declared in yaml file (intended to use in when training real model)
train:
	python src/train.py -p models/params/podcast/podcast_v2.yml

# sends the current project to the server (use from local computer)
send:
	bash -x utils/ssh/send.sh

# receives the logs, weights and outputs from server (use from local computer)
receive:
	bash -x utils/ssh/receive.sh

# builds the container where to perform training and synthesis
build-container:
	docker build -f utils/docker/Dockerfile -t jgarciapueyo/melnet .

# runs the docker container where to perform training and synthesis
run-container:
	bash utils/docker/melnet-run-container.sh