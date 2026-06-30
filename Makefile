PYTHON ?= python3
AGENT_PORT ?= 8090
RAG_PORT ?= 8091

.PHONY: install-dev test lint typecheck contracts contracts-check agent-eval rag-eval verify docker-build run-agent-runtime run-rag-runtime

install-dev:
	$(PYTHON) -m pip install -e "packages/shared[dev]"
	$(PYTHON) -m pip install -e "services/agent-runtime[dev]"
	$(PYTHON) -m pip install -e "services/rag-runtime[dev]"
	$(PYTHON) -m pip install -e "packages/sdk[dev]"

test:
	cd packages/shared && pytest
	cd services/agent-runtime && pytest
	cd services/rag-runtime && pytest
	cd packages/sdk && pytest

lint:
	ruff check packages/shared services/agent-runtime services/rag-runtime packages/sdk scripts

typecheck:
	cd packages/shared && mypy .
	cd services/agent-runtime && mypy .
	cd services/rag-runtime && mypy .
	cd packages/sdk && mypy .

contracts:
	$(PYTHON) scripts/generate_contracts.py

contracts-check:
	$(PYTHON) scripts/generate_contracts.py --check

agent-eval:
	cd services/agent-runtime && $(PYTHON) -m allcallall_agent_runtime.eval_runner --out evals/reports

rag-eval:
	cd services/rag-runtime && $(PYTHON) -m allcallall_rag_runtime.eval_runner --out evals/reports

verify: test lint typecheck contracts-check agent-eval rag-eval

docker-build:
	docker build -f services/agent-runtime/Dockerfile -t allcallall-agent-runtime:local .
	docker build -f services/rag-runtime/Dockerfile -t allcallall-rag-runtime:local .

run-agent-runtime:
	cd services/agent-runtime && uvicorn allcallall_agent_runtime.main:app --reload --port $(AGENT_PORT)

run-rag-runtime:
	cd services/rag-runtime && uvicorn allcallall_rag_runtime.main:app --reload --port $(RAG_PORT)
