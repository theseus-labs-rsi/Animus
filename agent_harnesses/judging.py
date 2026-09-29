"""工厂 judge 的适配层。

judge 真源始终是工厂的 `eval/judge.py`。合并后控制面与 judge 同仓库，但仍保留
按文件路径动态加载，而不是 `import eval.judge`，原因有两个：

1. 记录工厂 commit 与 judge.py 内容 hash 作为判分版本，判分口径必须可追溯；
2. 仓库根存在顶层 `eval` 包，与运行期 sys.path 上的同名包有冲突风险，按文件
   路径加载才能明确锁定唯一实现。

默认从本仓库根加载；experiment 的 `judge.factory_root` 与环境变量
`MEMORY_BENCH_FACTORY_ROOT` 仍可覆盖（用于跨仓库对照），正常情况下无需配置。
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .config import REPOSITORY_ROOT, ConfigurationError

FACTORY_ROOT_ENV = "MEMORY_BENCH_FACTORY_ROOT"
_JUDGE_RELPATH = Path("eval") / "judge.py"


def resolve_factory_root(judge_cfg: Mapping[str, Any] | None = None) -> Path:
    raw = str((judge_cfg or {}).get("factory_root") or "").strip()
    source = "experiment.judge.factory_root"
    if not raw:
        raw = os.environ.get(FACTORY_ROOT_ENV, "").strip()
        source = f"env {FACTORY_ROOT_ENV}"
    if not raw:
        # 合并后 judge 就在本仓库根，无需外部路径配置。
        raw = str(REPOSITORY_ROOT)
        source = "repository root"
    root = Path(raw).expanduser().resolve()
    if not (root / _JUDGE_RELPATH).is_file():
        raise ConfigurationError(f"{source}={root}: 未找到 {_JUDGE_RELPATH}")
    return root


def judge_version(root: Path, module: Any | None = None) -> dict[str, Any]:
    judge_file = root / _JUDGE_RELPATH
    sha = hashlib.sha256(judge_file.read_bytes()).hexdigest()[:16]
    commit = None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        commit = result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    model = None
    endpoint_host = None
    endpoint_fingerprint = None
    config = getattr(module, "config", None) if module is not None else None
    if config is not None:
        model = str(getattr(config, "JUDGE_MODEL", "") or "") or None
        base_url = str(getattr(config, "BASE_URL", "") or "").strip()
        if base_url:
            endpoint_host = urlsplit(base_url).hostname
            endpoint_fingerprint = hashlib.sha256(
                base_url.rstrip("/").encode("utf-8")
            ).hexdigest()[:16]
    return {
        "factory_root": str(root),
        "factory_commit": commit,
        "judge_sha256": sha,
        "judge_model": model,
        "judge_endpoint_host": endpoint_host,
        "judge_endpoint_fingerprint": endpoint_fingerprint,
    }


class Judge:
    """工厂 eval/judge.py 的薄包装，附带版本信息。"""

    def __init__(self, module: Any, version: dict[str, Any]):
        self._module = module
        self.version = version

    def judge_answer(self, q: dict, pred: str, use_llm: bool = True) -> bool:
        return bool(self._module.judge_answer(q, pred, use_llm=use_llm))

    def is_judgeable(self, q: dict) -> bool:
        return bool(self._module.is_judgeable(q))

    def judge_mode(self, q: dict) -> str | None:
        mode, _, _ = self._module.judge_spec(q)
        return mode

    def gold_display(self, q: dict) -> Any:
        return self._module.gold_display(q)

    def judge_l2_partial(self, q: dict, pred: str, use_llm: bool = True) -> float:
        return float(self._module.judge_l2_partial(q, pred, use_llm=use_llm))

    def classify_refusal(self, pred: str, lure: Any = None) -> str:
        return str(self._module.classify_refusal(pred, lure))


def load_judge(root: Path) -> Judge:
    """按文件路径加载工厂 judge 模块。

    不能用 `import eval.judge`：本仓库根目录也有 eval 包，会先被命中。
    judge.py 自己会把工厂根加入 sys.path 以 import 工厂 config。
    """
    judge_file = root / _JUDGE_RELPATH
    spec = importlib.util.spec_from_file_location("memory_bench_factory_judge", judge_file)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"无法加载工厂 judge 模块: {judge_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise ConfigurationError(f"工厂 judge 模块加载失败: {judge_file}: {exc}") from exc
    return Judge(module, judge_version(root, module))
