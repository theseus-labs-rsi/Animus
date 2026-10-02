"""Fail closed on a frozen terminal run before its manifest can be rewritten.

The independent terminal reconciliation proves history, not permission to
execute. A later explicit cumulative migration will need a separate validated
entry; ordinary factory and production routes must never infer that permission
from a copied run directory or an old successful stage flag.
"""
import json
from contextvars import ContextVar
from pathlib import Path

_bounded_wrapper_run = ContextVar("bounded_original_run", default=None)
_explicit_cumulative_run = ContextVar("explicit_cumulative_run", default=None)


def enter_bounded_run(run_id: str):
    """Called only after the wrapper installed its cumulative call gate."""
    return _bounded_wrapper_run.set(run_id)


def leave_bounded_run(token) -> None:
    _bounded_wrapper_run.reset(token)


def enter_explicit_cumulative_run(permit):
    """A dedicated entry must present the sealed read-only prewrite result."""
    from pipeline.cumulative_prewrite import is_valid_permit
    run_id = getattr(permit, "run_id", None)
    if not isinstance(run_id, str) or not is_valid_permit(permit, run_id):
        raise ValueError("A validated cumulative prewrite permit is required")
    require_bounded_run(run_id)
    return _explicit_cumulative_run.set(permit)


def leave_explicit_cumulative_run(token) -> None:
    _explicit_cumulative_run.reset(token)


def require_bounded_run(run_id: str) -> None:
    """A cumulative paid boundary must live inside its installed call gate."""
    if _bounded_wrapper_run.get() != run_id:
        raise ValueError("Cumulative review requires the active bounded call wrapper")


def check_before_run_init(manifest_path: Path) -> None:
    """Read-only check before Run.__init__ updates manifest.env."""
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cumulative = (manifest.get("config") or {}).get("cumulative_recovery")
    if manifest.get("terminal_reconciliation") is not None or cumulative is not None:
        from pipeline.cumulative_prewrite import is_valid_permit
        permit = _explicit_cumulative_run.get()
        if (cumulative is None or not isinstance(manifest.get("run_id"), str)
                or not manifest["run_id"] or not is_valid_permit(permit, manifest["run_id"])
                or permit.destination != manifest_path.parent.resolve()
                or _bounded_wrapper_run.get() != manifest.get("run_id")):
            raise ValueError("An explicit cumulative recovery needs its dedicated entry before factory initialization")
        return
    profile_path = manifest_path.with_name("experiment_profile.json")
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("stopped") is True:
            raise ValueError("A stopped cumulative budget ledger cannot enter the plain factory")
        if profile.get("source_drift"):
            raise ValueError("A source-drifted cumulative run cannot enter the plain factory")
        if _bounded_wrapper_run.get() != manifest_path.parent.name:
            raise ValueError("A cumulative run requires its active bounded wrapper before factory initialization")


def check_before_production(run) -> None:
    """Defend direct production callers before state, budget, or stage writes."""
    if run.manifest.get("terminal_reconciliation") is not None:
        raise ValueError("A terminal-derived run cannot skip its cumulative world-review control gate")
    if (run.manifest.get("config") or {}).get("cumulative_recovery") is not None:
        raise ValueError("An explicit cumulative recovery cannot enter production before review and repair acceptance")
    if run.has("experiment_profile.json"):
        profile = run.read("experiment_profile.json")
        if profile.get("stopped") is True or profile.get("source_drift"):
            raise ValueError("Production cannot bypass a stopped or drifted cumulative ledger")
        if _bounded_wrapper_run.get() != run.run_id:
            raise ValueError("Production cannot bypass the cumulative call wrapper")
