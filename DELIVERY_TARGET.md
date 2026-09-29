# 按交付量生成 benchmark

生成任务可以指定最终题量、能力线范围和语料 token 目标。系统在白皮书阶段分配各线题量，在世界完成前检查真实可枚举的题目，必要时请世界作者补充业务事实。最终报告保留原始订单、成题、审查状态、四选手得分完整度与简单题筛除记录。

`--delivery-target` 接收 JSON。例如：

```json
{
  "version": 1,
  "final_questions": 500,
  "count_stage": "selection_complete",
  "requested_lines": ["L1_timeline", "L2_relational", "L3_process", "L5_conflict", "L6_refusal", "L7_consolidation", "L8_transition"],
  "per_line_min": {},
  "per_line_max": {},
  "corpus_tokens": 1000000,
  "tokenizer": "cl100k_base@0.12.0",
  "max_supply_rounds": 2
}
```

`corpus_tokens` 逐篇计数正文，要求固定的 tokenizer 版本。语料报告记录每篇、每期和总量。新增草堆按剩余 token 缺口分批生成，已完成的正文与计数检查点可在同一运行中复用。

候选订单预算可用 `--question-budget` 明确指定。省略时，规划阶段按最终目标的三倍预留。已有四选手数据时，可通过 `--delivery-survival-rates` 传入逐线留存率，系统按来源、样本数和 1.25 倍余量计算候选储备：

```json
{
  "L1_timeline": {"rate": 0.42, "source": "实验报告及输入哈希", "sample_size": 100},
  "L2_relational": {"rate": 0.35, "source": "实验报告及输入哈希", "sample_size": 100}
}
```

证据文件须覆盖目标中的每条线。白皮书会记录不适用或尚无实现的能力线，交付报告逐线显示缺口。四选手成绩齐全后，筛选阶段剔除四者全对的题，再从剩余题中尽量均衡地选出目标数量；额外合格题留在筛选记录中。`per_line_min` 和 `per_line_max` 可约束最终分布。

运行到 `quality` 时，`10_delivery_target.json` 会先记录生成侧数量和语料 token。配置四选手评测并运行到 `selection` 后，报告更新正式交付数量。每次运行的输入和目标会冻结；改变目标请开新运行。

当前自动补量发生在世界完成前，依据真实结构候选缺口提出补充要求。四选手筛选后的缺口进入报告，后续追加供给需要新运行及独立预算。实际目标仍取决于世界结构、语料审查和选手成绩。
