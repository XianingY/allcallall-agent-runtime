SYSTEM_PYTHON ?= python3
VENV ?= .venv
PYTHON ?= $(abspath $(VENV)/bin/python)
AGENT_PORT ?= 8090
RAG_PORT ?= 8091

.PHONY: venv install-dev test lint typecheck contracts contracts-check agent-eval rag-eval portfolio-eval ai-agent-portfolio-eval verify docker-build run-agent-runtime run-rag-runtime

venv:
	@if [ ! -x "$(PYTHON)" ]; then $(SYSTEM_PYTHON) -m venv $(VENV); fi
	$(PYTHON) -m pip install --upgrade pip

install-dev: venv
	$(PYTHON) -m pip install -e "packages/shared[dev]"
	$(PYTHON) -m pip install -e "services/agent-runtime[dev]"
	$(PYTHON) -m pip install -e "services/rag-runtime[dev]"
	$(PYTHON) -m pip install -e "services/sandbox-runner[dev]"
	$(PYTHON) -m pip install -e "services/interview-mcp[dev]"
	$(PYTHON) -m pip install -e "packages/sdk[dev]"

test:
	cd packages/shared && $(PYTHON) -m pytest
	cd services/agent-runtime && $(PYTHON) -m pytest
	cd services/rag-runtime && $(PYTHON) -m pytest
	cd services/sandbox-runner && $(PYTHON) -m pytest
	cd services/interview-mcp && $(PYTHON) -m pytest
	cd packages/sdk && $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check packages/shared services/agent-runtime services/rag-runtime services/sandbox-runner services/interview-mcp packages/sdk scripts

typecheck:
	cd packages/shared && $(PYTHON) -m mypy .
	cd services/agent-runtime && $(PYTHON) -m mypy .
	cd services/rag-runtime && $(PYTHON) -m mypy .
	cd services/sandbox-runner && $(PYTHON) -m mypy .
	cd services/interview-mcp && $(PYTHON) -m mypy .
	cd packages/sdk && $(PYTHON) -m mypy .

contracts:
	$(PYTHON) scripts/generate_contracts.py

contracts-check:
	$(PYTHON) scripts/generate_contracts.py --check

agent-eval:
	cd services/agent-runtime && $(PYTHON) -m allcallall_agent_runtime.eval_runner --out evals/reports

rag-eval:
	cd services/rag-runtime && $(PYTHON) -m allcallall_rag_runtime.eval_runner --out evals/reports

portfolio-eval:
	$(PYTHON) scripts/portfolio_eval.py --out docs/generated-ai-agent-portfolio-eval

ai-agent-portfolio-eval: portfolio-eval

verify: test lint typecheck contracts-check agent-eval rag-eval

docker-build:
	docker build -f services/agent-runtime/Dockerfile -t allcallall-agent-runtime:local .
	docker build -f services/rag-runtime/Dockerfile -t allcallall-rag-runtime:local .

run-agent-runtime:
	cd services/agent-runtime && $(PYTHON) -m uvicorn allcallall_agent_runtime.main:app --reload --port $(AGENT_PORT)

run-rag-runtime:
	cd services/rag-runtime && $(PYTHON) -m uvicorn allcallall_rag_runtime.main:app --reload --port $(RAG_PORT)
