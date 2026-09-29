"""Track adapter 的最小公共接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import ExperimentConfig, SystemConfig
from ..planning import RunPlan


class TrackAdapter(ABC):
    """把统一 RunPlan 翻译为某条赛道的执行动作。"""

    def preflight(
        self,
        system: SystemConfig,
        experiment: ExperimentConfig,
        plan: RunPlan,
    ) -> dict:
        return {"system_id": system.system_id, "ok": True, "errors": []}

    @abstractmethod
    def build_command(
        self,
        experiment: ExperimentConfig,
        system: SystemConfig,
        plan: RunPlan,
    ) -> list[str]:
        """返回不经 shell 展开的 argv；未接入的对象应明确拒绝。"""
