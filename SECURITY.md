# Security Policy

## Supported Versions

The runtime is pre-1.0. Security fixes are made on the current default branch;
older snapshots and downstream forks are not maintained by this project.

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability. Send a private report
to `security@allcallall.com` with:

- the affected service, package, and commit SHA;
- reproduction steps or a minimal proof of concept;
- the expected impact and required configuration;
- any suggested mitigation or disclosure constraints.

Maintainers will coordinate triage, remediation, and disclosure with the
reporter. Timing depends on severity, reproducibility, and release risk; this
policy does not promise a fixed response or remediation window.

## Scope

In scope:

- authentication between Go and Python services;
- authorization context, approval boundaries, and write-tool proposals;
- prompt injection defenses, grounding, citations, and sensitive trace data;
- sandbox execution, checkpoint persistence, tool queues, and provider
  adapters;
- contracts, containers, CI, packaging, and release configuration.

Generally out of scope:

- reports without a reproducible security impact;
- public upstream dependency advisories without a runtime-specific exploit;
- model-quality disagreements that do not cross a documented safety boundary;
- attacks against a modified or incorrectly secured downstream deployment.

## Safe Harbor

Good-faith research is welcome when it avoids privacy violations, data loss,
service disruption, persistence, and access beyond what is needed to
demonstrate the issue. Allow maintainers a reasonable opportunity to remediate
before public disclosure.

For non-sensitive questions and ordinary bugs, use [SUPPORT.md](SUPPORT.md).
