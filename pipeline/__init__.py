"""
memory_bench_factory.pipeline — Benchmark 生成 pipeline。

按 W0 5 份 anchor 设计的多阶段管线:
  scenario_loader.py     — 加载 ScenarioSpec(W1 Day-1 提供 OfficeMem 接入)
  stage_a_dimensions.py  — Stage A: 推断 (I, S, V, T) 4 维
  stage_b_config.py      — Stage B: 推断 O 维度(失败模式 × 算子 配额)[后续]
  stage_c_corpus.py      — Stage C: 从 corpus_samples 扩展生成更多 corpus    [后续]
  stage_d_questions.py   — Stage D: 按 O 配额签发带标签题                   [后续]
  stage_e_output.py      — Stage E: 输出三重兼容 Benchmark JSON              [后续]

参考 docs/anchors/*.md。
"""
