"""
eval — W2 评测基础设施(V6 第一次建评测端)。

V1-V5 全在【生成端】打转(怎么造出 benchmark);V6 终于建【评测端】:
拿造好的 benchmark 去跑真实 baseline,用数据验证 benchmark 的科学价值。

模块:
  - memory_interface  记忆系统接口(MemorySystem ABC)+ EmbedMemory(DMXAPI dense 检索,机制同 simpleMem)
  - baseline_r1       R1 单轮 baseline(检索 top-k → LLM 合成答案,= SingleTurnAdaptor)
  - judge             判分(EM/子串 + LLM 兜底)+ M3 信号竞争触发分类
  - run_eval          编排:逐 period ingest → 终态提问 → 判分 → 聚合(标准 A + failmode 命中率)

详见 redesign_v6.md §1 标准 A(评测价值)/ §2 转变 2(信号竞争)。
"""
