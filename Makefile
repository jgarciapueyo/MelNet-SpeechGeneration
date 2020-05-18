.PHONY: clean-pyc data

# found in https://krzysztofzuraw.com/blog/2016/makefiles-in-python-projects.html
clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	name '*~' -exec rm --force  {}

data:
	python ./src/data/librispeech.py -r ./data
