.PHONY: style quality test

check_dirs := korquad_qg/ scripts/ tests/

style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

test:
	pytest

prepare-dataset:
	mkdir data
	wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -O data/train.json
	wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -O data/dev.json
