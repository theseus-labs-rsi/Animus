# Animus

<p>
  <a href="README.md"><img src="assets/readme/language/zh-active.svg" alt="中文（当前语言）" width="96" height="34"></a>
  <a href="README.en.md"><img src="assets/readme/language/en.svg" alt="Switch to English" width="96" height="34"></a>
</p>

Animus 从结构化 seed 和生成目标规划随时间演变的领域世界，先生成文书，再生成有材料依据的评测题，覆盖信息提取、知识更新、时间推理、多跳关系与来源冲突等能力。流程支持逐线供给规划、逐题审查、四模型试答和标准 benchmark 导出。

[试答结果](#试答结果) · [案例](#案例) · [快速开始](#快速开始)

## 试答结果

两张图汇总既有生成期筛选样本中的 240 道题，覆盖法律、金融、鉴证和保险四个领域。图中列出四选手成绩、七项能力表现和题量分布。

![240 道题的四选手总成绩、七项能力雷达图和四领域成绩](assets/readme/01_performance_dashboard.png)

![四领域与七项能力的题量分布、四选手逐能力正确数和准确率](assets/readme/02_coverage_and_capabilities.png)

## 案例

以下三题选自上述 240 题。题目与材料均由 Animus 生成；表格摘述关键记录，模型回答保留原文。对应的[原始材料与回答](examples/case_studies.json)可直接查看。

### 重开工单的历史结论

场景：法律／商户合规复核 · 能力：信息提取（IE）

> 第 11 周时，工单「甜心甜品屋台账补交重开复核」的复查结论是什么？

| 时间 | 关键记录摘要 |
|---|---|
| 第 5 周 · 2 月 3 日 | 原工单「材料待补复核」的结论为“待补交完整台账”。 |
| 第 7 周 · 2 月 17 日 | 从原工单派生出「台账补交重开复核」工单。 |
| 第 10 周 · 3 月 10 日 | 证照比对表记载甜心甜品屋许可证有效、待台账结论。 |
| 第 11 周 · 3 月 17 日 | 完整台账补交件到达，登记为待复核证据，复核结论另行安排。 |
| 第 13 周 · 3 月 31 日 | 复核记录回溯重开工单在 **3 月 10 日**的结论为“证照有效待台账结论”，并登记 3 月 31 日的新结论“台账已补齐、恢复评估中”。 |

答案：**证照有效待台账结论**。回答需要定位重开工单，并沿其记录还原第 11 周的状态。

| 模型 | 原始回答 | 结果 |
|---|---|---|
| GPT-5.6 | 证照有效待台账结论。 | 正确 |
| DS v4.1 | 证照有效待台账结论（待台账结论，第11周补交件到达但复核结论未更新，沿用3月10日值）。 | 正确 |
| GPT-5.3 | 待补证后重核 | 错误 |
| DS 0731 | 待补交完整台账 | 错误 |

DS 0731 的回答与原工单第 5 周的结论相同。

### 报告来源的切换时间

场景：保险／年度分析报告采用 · 能力：信息提取（IE）

> 第 6 周时，年度分析记录11 的采用来源是什么？

| 时间 | 关键记录摘要 |
|---|---|
| 第 5 周 · 2 月 3 日 | 采用变更单将年度分析记录11 的采用来源设为 S21。 |
| 第 6 周 · 2 月 10 日 | S22 到达并完成收件登记，台账保留既有采用来源。 |
| 第 7 周 · 2 月 17 日 | 新的采用变更单将该记录的来源改为 S22。 |

答案：**来源记录S21**。S22 的到达时间与正式采用时间相隔一周，第 6 周仍采用 S21。

| 模型 | 原始回答 | 结果 |
|---|---|---|
| GPT-5.6 | 来源记录S22。 | 错误 |
| DS v4.1 | 来源记录S21（客户11的FY2023/24暂定版，采用口径为暂定版口径）。 | 正确 |
| GPT-5.3 | 来源记录S21 | 正确 |
| DS 0731 | 来源记录S21 | 正确 |

### 证据充分性的首次变化

场景：法律／商户合规复核 · 能力：时间推理（TR）

> 工单「邻里超市价格标示复核」的证据充分性首次发生变化是在第几周？请回答期数。

| 时间 | 关键记录摘要 |
|---|---|
| 第 7 周 · 2 月 17 日 | 工单建立，证据充分性初始登记为“待补证”。 |
| 第 9 周 · 3 月 3 日 | 完整价签照片到达，登记为待复核证据。 |
| 第 12 周 · 3 月 24 日 | 复核完成，证据充分性更新为“已核验”。 |

答案：**第 12 周**。先确定第 7 周登记的初始值，再找到它首次变更的记录。

| 模型 | 原始回答 | 结果 |
|---|---|---|
| GPT-5.6 | 第12周。 | 正确 |
| DS v4.1 | 第12周 | 正确 |
| GPT-5.3 | 第12周。 | 正确 |
| DS 0731 | 第7周 | 错误 |

## 快速开始

需要 Python 3.10+；四模型试答还需要 Node.js 22.19.0+ 和 npm。在仓库根目录执行：

### 安装

Windows PowerShell：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -X utf8 -m pip install -r requirements-minimal.txt
Copy-Item .env.example .env
Copy-Item configs/env/secrets.env.example configs/env/secrets.env
```

<details>
<summary>Linux / macOS</summary>

```bash
python3 -m venv venv
./venv/bin/python -X utf8 -m pip install -r requirements-minimal.txt
cp .env.example .env
cp configs/env/secrets.env.example configs/env/secrets.env
```

</details>

安装试答用的 CLI：

```bash
npm install --global @openai/codex@0.153.2 @deepseek-ai/dsh@0.1.2-rc.1
```

在 `.env` 填写 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `MODEL`；`STRUCTURE_MODEL` 可单独指定世界设计模型。四模型试答使用 `configs/env/secrets.env` 中的接口和[模型配置](examples/release_four.json)，另见 [DSH profile](configs/dsh/README.md)。

### 生成与筛选

准备 seed JSON，格式见 [schema](schemas/seed_pack_v2.schema.json)、[转换指南](skills/realfiles-to-seedjson/SEEDJSON_GUIDE.md)和[合成示例](skills/realfiles-to-seedjson/examples/seed_v2_example.json)。seed 提供领域背景与约束；白皮书根据 seed、题量和语料目标设计实体、关系及跨期业务过程，再由程序检查并执行实例计划。

将交付目标保存为 `path/to/target.json`。以下 V1 目标要求四模型筛选后至少 200 题，总语料至少 100 万 token，并在列出的六条产线间均衡分配；`per_line_min`、`per_line_max` 可设置逐线题量边界。

```json
{
  "version": 1,
  "count_stage": "selection_complete",
  "final_questions": 200,
  "requested_lines": [
    "L1_timeline", "L2_relational", "L5_conflict",
    "L6_refusal", "L7_consolidation", "L8_transition"
  ],
  "per_line_min": {},
  "per_line_max": {},
  "corpus_tokens": 1000000,
  "tokenizer": "cl100k_base@0.12.0",
  "max_supply_rounds": 3
}
```

```powershell
.\venv\Scripts\python.exe -X utf8 tools/validate_seed_packs.py path/to/seed.json --require-generation-ready
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --seed-pack path/to/seed.json --delivery-target path/to/target.json --haystack-ratio 9 --semantic-workers 4 --release
```

系统先规划逐线供给，再生成世界、公开安排和出题订单；检查订单后生成正式材料与草堆，再完成题目表述、材料接地和质量审查。V1 默认预留筛后目标三倍的候选额度，并根据各线缺口补供给，轮数受 `max_supply_rounds` 限制。

`--release` 使用 `examples/release_four.json` 启用四模型试答，并按其配置剔除全员答对的题。只做生成与质量审查时，省略 `--release` 并加 `--to quality`。Linux/macOS 将解释器换成 `./venv/bin/python`。

语料 token 按文档正文、指定 tokenizer 和固定版本计数。上例的 `--haystack-ratio 9` 仍按字符指定草堆与其余正文至少 9:1；旧参数 `--target-mchars`（兼容别名 `--target-mtokens`）也按字符计量，不能与 `--delivery-target` 混用。

只生成语料和候选题时，使用 V2 目标。例如，下面的目标要求七线均分 200 道候选题，正式材料至少 10 万 token、总语料至少 100 万 token，草堆与正式材料的 token 比至少 9:1。白皮书根据这些目标设计世界规模和供给结构。

```json
{
  "version": 2,
  "count_stage": "generation_quality",
  "candidate_questions": 200,
  "requested_lines": [
    "L1_timeline", "L2_relational", "L3_process", "L5_conflict",
    "L6_refusal", "L7_consolidation", "L8_transition"
  ],
  "core_tokens": 100000,
  "corpus_tokens": 1000000,
  "filler_ratio": 9,
  "tokenizer": "cl100k_base@0.12.0",
  "max_supply_rounds": 3
}
```

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --seed-pack path/to/seed.json --delivery-target path/to/target.json --process-questions --semantic-workers 4
```

V2 默认运行到 `quality`，按 `04_questions.json` 中各线候选题计数；接地后的题量、质量资格和 L7 原生趋势分别验收。候选配额、语料实测和质量要求满足后，生产状态为 `generation_complete`；审查降级保留产物和警告。V2 不运行四模型筛选，CLI 会拒绝与 `--release` 或筛选阶段参数组合。需要筛后题量目标时使用 V1。

`LLM_CONCURRENCY` 限制实际在途模型请求，`--semantic-workers` 控制逐题语义审阅并行数（1–16）。单次请求的总截止和 HTTP 读取超时可在 `.env` 中设置 `LLM_DEADLINE_S`、`LLM_HTTP_READ_TIMEOUT_S`。完整参数见 `python -m pipeline.factory --help`。

### 续跑

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id>
```

续跑沿用 run 中冻结的 seed、目标和已完成阶段；阶段内检查点须通过输入及实现校验才会复用。改变 seed 或交付目标需新建 run。生产状态若为 `review_recovery_required`、`design_diagnosis_required` 或 `round_limit`，需先处理记录的原因再恢复。

已有 V1 完整生成结果时，补做试答与筛选：

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id> --release --from quality
```

`L3_process` 的过程题生成默认关闭，可用 `--process-questions` 启用。原生四模型评分当前尚不支持 `L3_process_trace`；上述筛选示例因此采用六线目标，含过程题的运行可停在 `quality`。

### 输出

结果保存在 `output/runs/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json`、`11_production.json` | 阶段状态、冻结配置与供给轮次 |
| `01_whitepaper.json`、`02_world.json` | 世界设计与执行后的业务事实 |
| `03_orders.json`、`04_questions.json` | 出题订单与候选题 |
| `05_corpus.json`、`05_corpus_token_scale.json` | 文档语料与正式材料、草堆的实际 token 数 |
| `06_grounded_questions.json`、`07_release.json` | 接地后题目、逐题审查结果和质量资格 |
| `08_calibration.json`、`09_selection.json` | 四模型试答状态与筛选结果 |
| `10_delivery_target.json` | 逐线题量、语料规模与交付目标的实测对照 |

审查降级会保留候选和警告，并反映在质量资格与交付报告中。`07_release.json` 的 `eligible` 记录质量资格；目标达成情况见 `10_delivery_target.json` 与 `11_production.json`。筛选得到可交付题目后，标准包位于该目录下的 `delivery/<attempt>/benchmark/`。

## 目录

| 路径 | 内容 |
|---|---|
| `pipeline/` | 世界、语料与题目生成，质量审查、筛选和导出 |
| `agent_harnesses/`、`eval/` | 模型运行器与评测系统接入 |
| `configs/`、`schemas/`、`skills/` | 运行配置、seed 格式与转换指南 |
| `examples/`、`assets/readme/` | 配置示例、案例数据与结果图 |
| [services/](services/README.md) | 可选网关、GPU 服务和多系统评测脚本 |
| `tools/`、`tests/` | 检查工具与测试 |

依赖分为[基础环境](requirements-minimal.txt)、[完整评测环境](requirements.txt)、[Mem0 接入](requirements-memory-mem0.txt)和[开发验证环境](requirements-test.txt)。按运行内容安装。安装开发验证环境后，可运行 `python -B tools/verify_repository.py` 执行隔离外部网络的回归检查，日志保存在 `output/repository_verification/`。
