# Contributing to AllCallAll Agent Runtime

Thank you for contributing to the Python Agent and RAG runtime.

## Before You Start

- Search existing issues and pull requests.
- Use public issues for reproducible bugs, focused feature proposals, and
  documentation problems.
- Report vulnerabilities privately through [SECURITY.md](SECURITY.md).
- Discuss changes to public imports, API routes, schemas, or the Go/Python
  ownership boundary before implementation.

## Development Setup

```bash
git clone git@github.com:XianingY/allcallall-agent-runtime.git
cd allcallall-agent-runtime
make install-dev
```

The main product repository is independent. Clone it as the sibling directory
`../AllCallAll` only when testing cross-repository integration.

## Making Changes

1. Create a focused branch from the current default branch.
2. Add or update tests before changing behavior.
3. Preserve published Python imports, HTTP routes, environment variables,
   JSON Schemas, and generated contracts.
4. Keep product writes in Go; Python write tools remain proposal-only.
5. Update canonical documentation and fixtures with behavior changes.
6. Use clear commit messages and push branches over SSH.

Never commit `.env`, `.omo`, `.workbuddy`, `output/`, credentials, traces with
private data, or generated evaluation output that was not intentionally
regenerated.

## Verification

```bash
make test
make lint
make typecheck
make docs-check
make contracts-check
```

Run `make verify` before requesting merge. Tests that require external MySQL or
provider credentials must remain explicitly opt-in and document their setup.

## Pull Requests

- Explain the runtime behavior being changed and why.
- List verification commands and any checks not run.
- Describe import, API, schema, environment, and cross-repository impact.
- Link evaluation methodology and fixture scope for metric changes.
- Call out security, approval, grounding, and privacy implications.

Contributions are licensed under the [MIT License](LICENSE), and participation
is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
