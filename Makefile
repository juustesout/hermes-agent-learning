.PHONY: test benchmark

test:
	bash scripts/run_tests.sh

benchmark:
	python3 test_adaptive_vs_standard.py
