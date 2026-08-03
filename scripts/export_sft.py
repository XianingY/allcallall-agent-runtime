"""CLI: export labeled badcases as an SFT JSONL dataset (Part 2).

Usage:
    python scripts/export_sft.py --db badcases.db --out evals/sft_dataset.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run as a plain script (editable install
# usually covers this, but the fallback keeps it robust in airgapped CI).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "agent-runtime"))

from allcallall_agent_runtime.badcase import BadcaseStore  # noqa: E402
from allcallall_agent_runtime.sft_dataset import export_sft_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export labeled badcases as an SFT JSONL dataset.")
    parser.add_argument("--db", default="badcases.db", help="BadcaseStore SQLite path")
    parser.add_argument("--out", default="evals/sft_dataset.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    store = BadcaseStore(args.db)
    records = store.list_sft_eligible()
    count = export_sft_dataset(records, Path(args.out))
    print(f"exported {count} SFT samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
