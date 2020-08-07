.PHONY: clean-pyc data

# found in https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

# starts the program to download the librispeech dataset to MelNet-SpeechGeneration/data/
data-librispeech:
	python src/data/librispeech.py -r datasets/librispeech

data-ljspeech:
	python src/data/ljspeech.py -r datasets/ljspeech

# starts training on small MelNet model to test locally if training runs correctly
dummy-train:
	python src/train.py -p models/params/podcast/dummymodel_podcast.yml

# starts training on MelNet declared in yaml file (intended to use in when training real model)
train:
	#python src/train.py -p models/params/podcast/podcast_v3.yml --tier 2
	python src/train.py -p models/params/ljspeech/ljspeech_v1.yml --tier 4

# performs synthesis
synthesis:
	#python src/synthesis.py -p models/params/librispeech/librispeech_v2.yml -s models/params/librispeech/synthesis_librispeech_v2.yml -t300
	python src/synthesis.py -p models/params/ljspeech/ljspeech_v1.yml -s models/params/ljspeech/synthesis_ljspeech_v1.yml -t200

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