from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
AGENT = ROOT / "services" / "agent-runtime"
RAG = ROOT / "services" / "rag-runtime"
SCHEMA_DIR = ROOT / "contracts" / "schemas"


def load_models(service_root: Path, package_name: str) -> Any:
    sys.path.insert(0, str(PACKAGES))
    sys.path.insert(0, str(service_root))
    try:
        return importlib.import_module(package_name + ".models")
    finally:
        sys.path = [item for item in sys.path if item not in {str(PACKAGES), str(service_root)}]


def clear_app_modules() -> None:
    prefixes = ("allcallall_agent_runtime.", "allcallall_rag_runtime.")
    for name in list(sys.modules):
        if (
            name in {"allcallall_agent_runtime", "allcallall_rag_runtime"}
            or name.startswith(prefixes)
        ):
            del sys.modules[name]


def schema_for(module: Any, class_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in class_names:
        model = getattr(module, name)
        out[name] = model.model_json_schema()
    return out


def build_contracts() -> dict[str, dict[str, Any]]:
    clear_app_modules()
    agent_models = load_models(AGENT, "allcallall_agent_runtime")
    agent = schema_for(
        agent_models,
        [
            "MeetingBriefRequest",
            "MeetingBriefResponse",
            "TraceEvent",
            "Citation",
            "ToolProposal",
            "EvidencePack",
            "ContextSufficiency",
            "IntentRoute",
            "RouteDecision",
            "CriticResult",
            "AgentHarnessMetadata",
            "LoopBudget",
            "LoopSpec",
            "LoopStep",
            "LoopTrace",
            "GraphExpansion",
            "MemoryReflection",
            "RiskAssessment",
        ],
    )
    clear_app_modules()
    rag_models = load_models(RAG, "allcallall_rag_runtime")
    rag = schema_for(
        rag_models,
        [
            "RetrievalQueryRequest",
            "RetrievalQueryResponse",
            "RerankRequest",
            "RerankResponse",
            "AgenticRetrievalRequest",
            "AgenticRetrievalResponse",
            "GroundingCheckRequest",
            "GroundingCheckResponse",
            "RetrievalRoute",
            "GraphExpansion",
        ],
    )
    return {"agent-runtime.schema.json": agent, "rag-runtime.schema.json": rag}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contracts = build_contracts()
    failed = False
    for filename, payload in contracts.items():
        path = SCHEMA_DIR / filename
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                print(f"contract out of date: {path}", file=sys.stderr)
                failed = True
        else:
            write_json(path, payload)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
