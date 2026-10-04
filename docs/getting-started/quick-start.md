# Quick Start

This guide starts both Python runtime services for local development. Product
clients and the Go backend are maintained in the sibling
[AllCallAll repository](https://github.com/XianingY/allcallall).

## Prerequisites

- Python 3.11 or newer; CI uses Python 3.12.
- A local checkout of this repository.
- The AllCallAll repository checked out as a sibling when testing end-to-end integration.

## Install and Verify

Run setup from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
make install-dev
make verify
```

## Run the Services

Use separate terminals from the repository root:

```bash
make run-agent-runtime
```

```bash
make run-rag-runtime
```

The Agent Runtime listens on port `8090` and the RAG Runtime listens on port
`8091` by default. Health endpoints are available at `/health`.

## Connect the Product Backend

The minimal environment settings and Compose integration are documented in
[AllCallAll Integration](../guides/allcallall-integration.md). Review the
[Configuration Reference](../reference/configuration.md) before enabling model
providers, checkpoint persistence, or external retrieval adapters.

## Next Steps

- [Architecture Overview](../architecture/overview.md)
- [Tool Bridge Protocol](../reference/tool-bridge-protocol.md)
- [Evaluation Methodology](../evaluation/methodology.md)
