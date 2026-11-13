PYTHON ?= python3
.PHONY: build test demo evaluate test-r sanitizers
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel 2

test: build
	ctest --test-dir build --output-on-failure
	$(PYTHON) -m pytest -q

demo: build
	$(PYTHON) agent.py --mode demo

evaluate: build
	$(PYTHON) evaluation/evaluate.py --engine build/qfa_engine --output-dir runs/evaluation

test-r: evaluate
	Rscript evaluation/check_reference.R --prices data/synthetic_prices.csv --output-dir runs/evaluation

sanitizers:
	cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DQFA_SANITIZERS=ON
	cmake --build build-sanitize --parallel 2
	ctest --test-dir build-sanitize --output-on-failure
