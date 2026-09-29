# Animus

<p>
  <a href="README.md"><img src="assets/readme/language/zh-active.svg" alt="中文（当前语言）" width="96" height="34"></a>
  <a href="README.en.md"><img src="assets/readme/language/en.svg" alt="Switch to English" width="96" height="34"></a>
</p>

Animus 从结构化 seed 生成随时间演变的领域世界、文书和评测题，覆盖信息提取、知识更新、时间推理、多跳关系与来源冲突等能力。生成流程支持逐题审查、四模型试答和标准 benchmark 导出。

[试答结果](#试答结果) · [案例](#案例) · [快速开始](#快速开始)

## 试答结果

两张图汇总生成期筛选样本中的 240 道题，覆盖法律、金融、鉴证和保险四个领域。图中列出四选手成绩、七项能力表现和题量分布。

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

在 `.env` 配置生成与裁判接口，在 `configs/env/secrets.env` 配置四模型接口。另见[模型配置](examples/release_four.json)与 [DSH profile](configs/dsh/README.md)。

### 生成与筛选

准备 seed JSON，格式见 [schema](schemas/seed_pack_v2.schema.json) 和[转换指南](skills/realfiles-to-seedjson/SEEDJSON_GUIDE.md)。

```powershell
.\venv\Scripts\python.exe -X utf8 tools/validate_seed_packs.py path/to/seed.json --require-generation-ready
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --seed-pack path/to/seed.json --min-questions 200 --target-mchars 1 --haystack-ratio 9 --semantic-workers 4 --release
```

`--release` 启用四模型试答，并剔除全员答对的题；省略时运行生成流程。`--target-mchars 1` 指定 100 万字符的正文目标，`--haystack-ratio 9` 指定草堆与其余正文至少 9:1。Linux/macOS 将解释器换成 `./venv/bin/python`。

### 续跑

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id>
```

已有完整生成结果时，补做试答与筛选：

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id> --release --from quality
```

完整参数见 `python -m pipeline.factory --help`。发布配置默认关闭 `L3_process_trace` 过程题，其过程语义评分尚未接入。

### 输出

结果保存在 `output/runs/<run_id>/`：`07_release.json` 记录质量审查，`08_calibration.json` 记录试答状态，`09_selection.json` 记录筛选结果。标准包位于该目录下的 `delivery/<attempt>/benchmark/`。

## 目录

| 路径 | 内容 |
|---|---|
| `pipeline/` | 世界、语料与题目生成，质量审查、筛选和导出 |
| `agent_harnesses/`、`eval/` | 模型运行器与评测系统接入 |
| `configs/`、`schemas/`、`skills/` | 运行配置、seed 格式与转换指南 |
| `examples/`、`assets/readme/` | 配置示例、案例数据与结果图 |
| [services/](services/README.md) | 可选网关、GPU 服务和多系统评测脚本 |
| `tools/`、`tests/` | 检查工具与测试 |

依赖分为[基础环境](requirements-minimal.txt)、[完整评测环境](requirements.txt)和 [Mem0 接入](requirements-memory-mem0.txt)。按运行内容安装。
