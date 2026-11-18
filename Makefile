.PHONY: test benchmark install-plugin check-plugin-sync

test:
	bash scripts/run_tests.sh

benchmark:
	python3 test_adaptive_vs_standard.py

install-plugin:
	python3 plugin_sync.py install

check-plugin-sync:
	python3 plugin_sync.py check
