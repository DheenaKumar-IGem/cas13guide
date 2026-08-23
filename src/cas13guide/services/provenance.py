"""Append-only provenance ledger.

Every index, model, parameter, exclusion and uncertainty decision is recorded
here so a result can be reproduced and diffed. The ledger is part of the
output, not a debug aid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    kind: str
    name: str
    version: str
    detail: dict[str, Any] = field(default_factory=dict)


class ProvenanceLedger:
    def __init__(self, spec_fingerprint: str) -> None:
        self.spec_fingerprint = spec_fingerprint
        self._records: list[ProvenanceRecord] = []

    def record(self, kind: str, name: str, version: str, **detail: Any) -> None:
        self._records.append(ProvenanceRecord(kind=kind, name=name, version=version, detail=detail))

    def exclusion(self, what: str, why: str, **detail: Any) -> None:
        """Log something deliberately left out. Silent exclusions are the bug."""
        self.record("exclusion", what, version="-", why=why, **detail)

    @property
    def records(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(self._records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_fingerprint": self.spec_fingerprint,
            "records": [asdict(r) for r in self._records],
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
