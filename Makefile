.PHONY: install test lint prepare train evaluate audit gradcam

install:
	python3 -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

prepare:
	python3 -m covid_xray prepare

train:
	python3 -m covid_xray train

evaluate:
	python3 -m covid_xray evaluate

audit:
	python3 -m covid_xray audit

gradcam:
	python3 -m covid_xray gradcam
