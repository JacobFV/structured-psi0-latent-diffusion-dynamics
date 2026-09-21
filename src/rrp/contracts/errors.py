"""Typed errors: every exception carries a stable code and an operator-readable message."""
from __future__ import annotations


class RRPError(Exception):
    code = "rrp_error"

    def __init__(self, message: str, *, code: str | None = None, **context):
        super().__init__(message)
        if code:
            self.code = code
        self.context = context

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "context": self.context}


class ContractError(RRPError, ValueError):
    code = "contract_violation"


class VersionConflict(RRPError):
    code = "version_conflict"


class ProvenanceError(RRPError, ValueError):
    code = "provenance_invalid"


class PrivilegedLeakError(RRPError):
    code = "privileged_leak"


class ControllerRejection(RRPError, ValueError):
    code = "controller_rejected"


class StaleActionError(ControllerRejection):
    code = "stale_action_chunk"
