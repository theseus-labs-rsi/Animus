"""Native Agent Track adapter。"""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import ConfigurationError, ExperimentConfig, SystemConfig
from ..planning import RunPlan
from ..runners.native_cli import preflight_system
from .base import TrackAdapter


class NativeTrackAdapter(TrackAdapter):
    def preflight(
        self,
        system: SystemConfig,
        experiment: ExperimentConfig,
        plan: RunPlan,
    ) -> dict:
        if system.runner == "builtin_diagnostic":
            return {"system_id": system.system_id, "ok": True, "errors": []}
        if system.runner == "native_cli":
            report = preflight_system(system, plan.model)
            if "timeout_s" in plan.execution:
                timeout = plan.execution["timeout_s"]
                if type(timeout) is not int or timeout <= 0:
                    report.setdefault("errors", []).append("execution.timeout_s 必须是正整数")
                    report["ok"] = False
                else:
                    report.setdefault("runtime_options", {})["timeout_s"] = timeout
            return report
        return {
            "system_id": system.system_id,
            "ok": False,
            "errors": [f"Native adapter 尚未接入 runner={system.runner!r}"],
        }

    def build_command(
        self,
        experiment: ExperimentConfig,
        system: SystemConfig,
        plan: RunPlan,
    ) -> list[str]:
        if system.runner == "native_cli":
            command = [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "agent_harnesses.runners.native_cli",
                "--plan",
                str(Path(plan.output_dir) / "run_plan.json"),
                "--run",
                str(Path(plan.benchmark["path"])),
                "--out",
                plan.output_dir,
            ]
            limit = int(experiment.execution.get("limit", 0))
            if limit:
                command.extend(["--limit", str(limit)])
            # 默认清理 workspace/runtime 临时区；execution.keep_scratch=true 时保留以便排障。
            if experiment.execution.get("keep_scratch"):
                command.append("--keep-scratch")
            return command
        if system.runner != "builtin_diagnostic":
            raise ConfigurationError(
                f"{system.system_id}: Native adapter 尚未接入 runner={system.runner!r}"
            )
        if system.role != "diagnostic":
            raise ConfigurationError("builtin_diagnostic 只能用于 diagnostic")
        command = [
            sys.executable,
            "-m",
            "agent_harnesses.runners.diagnostic",
            "--run",
            str(Path(plan.benchmark["path"])),
            "--out",
            plan.output_dir,
        ]
        limit = int(experiment.execution.get("limit", 0))
        if limit:
            command.extend(["--limit", str(limit)])
        return command
