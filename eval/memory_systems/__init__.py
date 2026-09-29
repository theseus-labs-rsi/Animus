"""
eval.memory_systems — 可插拔记忆系统工厂。

make_system("A") / make_system("simpleMem") → SimpleMem 实例
make_system("B") / make_system("fullcontext") → FullContext 实例
make_system("C") / make_system("iterative") → Iterative 实例
外部系统(mem0/zep/memos/amem)注册后同理。
"""
from eval.memory_systems.base import MemorySystem

_ALIASES = {
    "A": "simplemem",
    "B": "fullcontext",
    "C": "iterative",
}

_REGISTRY = {}

# 所有实现都懒加载：preflight 某个外部 adapter 时不得顺带 import factory config、
# embedding 或其它系统的 SDK。这样缺失的是目标依赖，而不是无关系统的全局配置。
_LAZY = {
    "simplemem": ("eval.memory_systems.simplemem", "SimpleMem"),
    "fullcontext": ("eval.memory_systems.fullcontext", "FullContext"),
    "iterative": ("eval.memory_systems.iterative", "Iterative"),
    "mem0": ("eval.memory_systems.mem0_adapter", "Mem0Adapter"),
    "zep": ("eval.memory_systems.zep_adapter", "ZepAdapter"),
    "memos": ("eval.memory_systems.memos_adapter", "MemOSAdapter"),
    "amem": ("eval.memory_systems.amem_adapter", "AMemAdapter"),
}


def canonical_system_name(name: str) -> str:
    """Validate a system name without loading adapters or contacting providers."""
    key = _ALIASES.get(name.upper(), name.lower())
    if key not in _REGISTRY and key not in _LAZY:
        raise ValueError(f"未知记忆系统: {name!r}")
    return key


def _resolve_class(name: str) -> type[MemorySystem]:
    key = canonical_system_name(name)
    cls = _REGISTRY.get(key)
    if cls is None and key in _LAZY:
        mod_path, cls_name = _LAZY[key]
        import importlib
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        _REGISTRY[key] = cls
    if cls is None:
        available = sorted(set(list(_REGISTRY.keys()) + list(_LAZY.keys())))
        raise ValueError(f"未知记忆系统: {name!r}  (可用: {available})")
    return cls


def make_system(name: str, **kwargs) -> MemorySystem:
    """按名称创建记忆系统实例。
    name: "A"/"B"/"C" 或 "simplemem"/"fullcontext"/"iterative"/"mem0"/"zep"/"memos"/"amem"。
    kwargs: 透传给构造函数(如 top_k, embed_mem, budget)。"""
    return _resolve_class(name)(**kwargs)


def preflight_system(name: str, **kwargs) -> dict:
    """Run an adapter-owned, side-effect-free preflight check."""
    result = _resolve_class(name).preflight(**kwargs)
    if not isinstance(result, dict):
        raise TypeError(f"{name} preflight 必须返回 dict")
    return result


def register(name: str, cls: type) -> None:
    """注册外部系统 adapter。"""
    _REGISTRY[name.lower()] = cls
