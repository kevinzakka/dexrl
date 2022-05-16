SHELL := /bin/bash

.PHONY: help check format
.DEFAULT: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  check: Run type checking and code styling *without* modifying files"
	@echo "  format: Run type checking and code styling inplace"

check:
	isort --check .
	black --check .
	flake8 --show-source .
	mypy .

format:
	isort .
	black .
	flake8 .
	mypy .
