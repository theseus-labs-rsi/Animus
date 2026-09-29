# Animus

<p>
  <a href="README.md"><img src="assets/readme/language/zh.svg" alt="切换到中文" width="96" height="34"></a>
  <a href="README.en.md"><img src="assets/readme/language/en-active.svg" alt="English (current language)" width="96" height="34"></a>
</p>

Animus generates evolving domain worlds, documents, and evaluation questions from structured seeds. It covers information extraction, knowledge updates, temporal reasoning, multi-hop relations, and source conflicts. The pipeline supports question review, evaluation with four models, and export to a standard benchmark package.

[Results](#results) · [Examples](#examples) · [Quick start](#quick-start)

## Results

These charts summarize a selected sample of 240 questions from the generation runs, covering legal, finance, forensic, and insurance domains. They show results for four model configurations, seven capability categories, and the question distribution.

![Overall results, seven capability profiles, and results by domain](assets/readme/01_performance_dashboard.png)

![Question distribution and model results by capability](assets/readme/02_coverage_and_capabilities.png)

## Examples

These three questions come from the same 240-question sample. Animus generated the questions and source documents. The timelines summarize the relevant records; model answers retain their original Chinese wording. The [source documents and answers](examples/case_studies.json) are included.

### A reopened case and its historical conclusion

Scenario: Legal / merchant compliance review · Capability: Information extraction (IE)

> In week 11, what was the review conclusion for the case “Sweetheart Dessert Shop — reopened review of supplemental ledgers”?

| Time | Relevant records, summarized |
|---|---|
| Week 5 · February 3 | The original case, “review of missing materials,” concludes that complete ledgers are still required. |
| Week 7 · February 17 | A new case for the supplemental ledgers is opened and linked to the original case. |
| Week 10 · March 10 | The license comparison record states that the shop's license is valid and a ledger conclusion is pending. |
| Week 11 · March 17 | Complete supplemental ledgers arrive and are registered as evidence awaiting review. The review conclusion is scheduled separately. |
| Week 13 · March 31 | A review record recalls the reopened case's **March 10** conclusion, “license valid; ledger conclusion pending,” and records its new March 31 conclusion, “ledgers complete; reinstatement under assessment.” |

Answer: **证照有效待台账结论** — license valid; ledger conclusion pending. The question requires identifying the reopened case and reconstructing its state in week 11.

| Model | Original answer | Result |
|---|---|---|
| GPT-5.6 | 证照有效待台账结论。 | Correct |
| DS v4.1 | 证照有效待台账结论（待台账结论，第11周补交件到达但复核结论未更新，沿用3月10日值）。 | Correct |
| GPT-5.3 | 待补证后重核 | Incorrect |
| DS 0731 | 待补交完整台账 | Incorrect |

DS 0731's answer matches the original case's week 5 conclusion, “complete ledgers still required.” GPT-5.3 answers “awaiting additional evidence before another review.”

### When a report becomes the adopted source

Scenario: Insurance / annual analysis report adoption · Capability: Information extraction (IE)

> In week 6, what was the adopted source for annual analysis record 11?

| Time | Relevant records, summarized |
|---|---|
| Week 5 · February 3 | An adoption record sets the source for annual analysis record 11 to S21. |
| Week 6 · February 10 | S22 arrives and is registered. The receipt log preserves the existing adopted source. |
| Week 7 · February 17 | A new adoption record changes the source to S22. |

Answer: **来源记录S21** — source record S21. S22 arrives one week before its formal adoption, so the adopted source in week 6 remains S21.

| Model | Original answer | Result |
|---|---|---|
| GPT-5.6 | 来源记录S22。 | Incorrect |
| DS v4.1 | 来源记录S21（客户11的FY2023/24暂定版，采用口径为暂定版口径）。 | Correct |
| GPT-5.3 | 来源记录S21 | Correct |
| DS 0731 | 来源记录S21 | Correct |

### The first change in evidence sufficiency

Scenario: Legal / merchant compliance review · Capability: Temporal reasoning (TR)

> In which week did evidence sufficiency first change for the case “Neighborhood Supermarket — price-label review”? Answer with the period number.

| Time | Relevant records, summarized |
|---|---|
| Week 7 · February 17 | The case is opened with evidence sufficiency initially recorded as “awaiting additional evidence.” |
| Week 9 · March 3 | Complete price-label photos arrive and are registered as evidence awaiting review. |
| Week 12 · March 24 | The review is completed and evidence sufficiency changes to “verified.” |

Answer: **Week 12**. Establish the initial value in week 7, then locate its first recorded change.

| Model | Original answer | Result |
|---|---|---|
| GPT-5.6 | 第12周。 | Correct |
| DS v4.1 | 第12周 | Correct |
| GPT-5.3 | 第12周。 | Correct |
| DS 0731 | 第7周 | Incorrect |

The first three answers say “week 12”; DS 0731 answers “week 7.”

## Quick start

Requires Python 3.10+. Evaluation with four models also requires Node.js 22.19.0+ and npm. Run commands from the repository root.

### Install

Windows PowerShell:

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

Install the evaluation CLIs:

```bash
npm install --global @openai/codex@0.153.2 @deepseek-ai/dsh@0.1.2-rc.1
```

Configure generation and judge endpoints in `.env`, and the four model endpoints in `configs/env/secrets.env`. See the [model configuration](examples/release_four.json) and [DSH profile](configs/dsh/README.md).

### Generate and select

Prepare a seed JSON using the [schema](schemas/seed_pack_v2.schema.json) and [conversion guide](skills/realfiles-to-seedjson/SEEDJSON_GUIDE.md).

```powershell
.\venv\Scripts\python.exe -X utf8 tools/validate_seed_packs.py path/to/seed.json --require-generation-ready
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --seed-pack path/to/seed.json --min-questions 200 --target-mchars 1 --haystack-ratio 9 --semantic-workers 4 --release
```

`--release` runs the four model configurations and removes questions all four answered correctly. Omitting it runs the generation pipeline. `--target-mchars 1` sets a target of one million body-text characters; `--haystack-ratio 9` sets a minimum ratio of 9:1 between haystack and other body text. On Linux/macOS, use `./venv/bin/python` as the interpreter.

### Resume

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id>
```

For a completed generation run, add model evaluation and selection:

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id> --release --from quality
```

See `python -m pipeline.factory --help` for all options. Release configurations disable `L3_process_trace` questions by default; semantic scoring for process traces is pending integration.

### Output

Results are saved in `output/runs/<run_id>/`: `07_release.json` records quality review, `08_calibration.json` records evaluation status, and `09_selection.json` records selection results. The standard package is under `delivery/<attempt>/benchmark/` within that run directory.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | World, document, and question generation; quality review, selection, and export |
| `agent_harnesses/`, `eval/` | Model runners and evaluation system integrations |
| `configs/`, `schemas/`, `skills/` | Runtime configuration, seed formats, and conversion guides |
| `examples/`, `assets/readme/` | Configuration examples, case data, and result charts |
| [services/](services/README.en.md) | Optional gateways, GPU services, and evaluation scripts |
| `tools/`, `tests/` | Inspection tools and tests |

Dependency lists cover the [core environment](requirements-minimal.txt), [full evaluation environment](requirements.txt), and [Mem0 integration](requirements-memory-mem0.txt). Install the set needed for your workflow.
