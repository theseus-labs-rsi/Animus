"""Pure formatting shared by FullContext execution and no-provider preflight."""


def header(sid, date) -> str:
    return f"[周期{sid} | 日期 {date}] "


def build_full_context(docs: list, budget: int = 120_000) -> tuple:
    blocks = [header(sid, date) + content for sid, date, content in docs]
    full = "\n\n".join(blocks)
    if len(full) <= budget:
        return full, False, len(full)
    kept, used = [], 0
    for block in blocks:
        if used + len(block) + 2 > budget:
            break
        kept.append(block)
        used += len(block) + 2
    return "\n\n".join(kept), True, used
