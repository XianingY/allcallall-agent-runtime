# Reference MCP Service

This service is a small, deterministic MCP implementation used to validate the
AllCallAll sandbox transport, authentication, read-tool, and idempotent
write-tool boundaries. It is reference infrastructure, not a production system
of record; durable AllCallAll product writes remain owned by the Go backend.

## Run Locally

Install dependencies from the repository root, then start the HTTPS endpoint:

```bash
make install-dev
cd services/reference-mcp
../../.venv/bin/python -m uvicorn allcallall_reference_mcp.main:app \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-keyfile /run/interview-tls/interview-mcp.key \
  --ssl-certfile /run/interview-tls/interview-mcp.crt
```

The service exposes `GET /health`, `GET /metrics`, and the streamable HTTP MCP
endpoint at `/mcp`.

## Compatibility and Security

The distribution is `allcallall-reference-mcp`. The former
`allcallall_interview_mcp` Python package remains available as an explicit
compatibility layer for one deprecation cycle and resolves to the same MCP
objects as `allcallall_reference_mcp`.

Existing deployments may continue using `INTERVIEW_MCP_DB_PATH`,
`INTERVIEW_MCP_BEARER_TOKEN_FILE`, `INTERVIEW_MCP_BEARER_TOKEN`, and
`MCP_INTERVIEW_TRUSTED_HOSTS`. TLS is mandatory for sandbox access, and trusted
hosts must remain exact DNS names rather than wildcard or suffix matches. See
the [configuration reference](../../docs/reference/configuration.md).
