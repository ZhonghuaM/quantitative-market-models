.PHONY: install test analysis clean cpp js

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

analysis:
	python scripts/run_analysis.py

cpp:
	c++ -std=c++17 -O3 cpp/monte_carlo_option.cpp -o cpp/monte_carlo_option
	./cpp/monte_carlo_option

js:
	node javascript/option_pricer.js

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -f cpp/monte_carlo_option
