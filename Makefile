.PHONY: setup test clean run-exploration

# Detect if running in a virtual environment
IN_VENV := $(shell python3 -c 'import sys; print(sys.prefix != sys.base_prefix)' 2>/dev/null)

# Define commands based on environment
ifeq ($(IN_VENV),True)
    PYTHON_CMD = python3
    PIP_CMD = pip
else
    # If not in venv, assume uv usage or standard python
    # We prefer using the created .venv if it exists
    ifneq (,$(wildcard .venv/bin/python))
        PYTHON_CMD = .venv/bin/python
        PIP_CMD = .venv/bin/pip
    else
        PYTHON_CMD = python3
        PIP_CMD = pip
    endif
endif

setup:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Creating virtual environment with uv..."
	uv venv
	@echo "Installing dependencies with uv..."
	uv pip install -r requirements.txt

test:
	$(PYTHON_CMD) -m unittest discover tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv

run-exploration:
	jupyter notebook notebooks/exploration.ipynb

# Run FastAPI service (loads weights from MODEL_PATH or models/semantic_classifier.pkl)
run-api:
	$(PYTHON_CMD) -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
