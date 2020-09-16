.PHONY: clean-pyc data

# found in https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

# builds the container where to perform training and synthesis
build-container:
	docker build -f utils/docker/Dockerfile -t melnet .

# runs the docker container where to perform training and synthesis
run-container:
	bash utils/docker/melnet-run-container.sh

# starts the program to download the librispeech dataset to MelNet-SpeechGeneration/data/
data-librispeech:
	python src/data/librispeech.py -r datasets/librispeech

# starts the program to download the ljspeech dataset to MelNet-SpeechGeneration/data/
data-ljspeech:
	python src/data/ljspeech.py -r datasets/ljspeech

# starts training a MelNet model
train-template:
	python src/train.py -p models/params/template_training.yml
	# If we wanted to train only the second tier of that model:
	# python src/train.py -p models/params/template_training.yml --tier 2
	# If we wanted to resume training of second tier from a checkpoint:
	# python src/train.py -p models/params/template_training.yml --tier 2 --checkpoint-path models/chkpt/ddataset_t6_l12.5.4.3.2.2_hd512_gmm10/tier2_20200101-000000.pt

# performs synthesis and generates spectrogram
synthesis-template:
	python src/synthesis.py -p models/params/template_training.yml -s models/params/template_synthesis.yml -t 200

# sends the current project to the server (use from local computer)
# TODO: add README in utils/ssh
send:
	bash -x utils/ssh/send.sh

# receives the logs, weights and outputs from server (use from local computer)
receive:
	bash -x utils/ssh/receive.sh