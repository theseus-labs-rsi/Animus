# Seed JSON 字段说明

这份说明解释 `realfiles-to-seedjson` skill 生成的 JSON 如何组织信息，以及这些信息如何在后续指导白皮书和结构化世界生成。

## 项目接入约定

新输出采用整数 `schema_version: 2`。共享格式定义位于 `../../schemas/seed_pack_v2.schema.json`，运行入口使用 `pipeline.seed_pack.validate_seed_pack()`；旧版 v1 种子继续可加载。

以下 JSON 展示字段片段。完整可加载示例见 `examples/seed_v2_example.json`；`examples/source_workspace/` 提供三份完全合成的资料，示例保存了这些文件的实际 SHA-256，可离线核验。

- 必填顶层字段沿用旧版：`schema_version`、`seed_id`、`family`、`title`、`description`、`sources`、`task`、`exemplars`、`mechanisms`、`blueprint_requirements`。
- 原始 v2 种子额外要求 `document_map`、`generation_contract`、`review`，缺少对应内容时显式使用空数组或对象。原子事实、详细读取记录和未决冲突放入审查记录；经确认的业务要求放入生成字段。内部已绑定哈希的生成视图可省略审计字段，不能用该内部格式替代原始交付。
- `sources.role` 记录使用权限，`sources.source_kind` 记录资料性质。生成字段的 `source_refs` 仅能引用 `corpus` 或 `builder_only` 来源，原文示例仅能引用 `corpus`。
- `review.blocking_issues` 非空时允许保存和检查草稿，生成入口暂停使用该种子。`quality_status` 为审查记录中的声明，不能代替校验结果。
- 文件路径按原始资料工作区解释；离线核验时指定 `--source-root`，默认仍按仓库根目录解释。
- 状态条件、指标期间、未决项和合成设计依据继续交给原 LLM 作者与审阅者理解。代码校验结构、引用、权限和版本。
- 可执行实体、字段、关系、事件、因果规则和机制标记 `conditional` 或 `unresolved` 时，允许保存草稿，生成入口拒绝使用。先确认其结构要求，或将未决语义边界移入 `generation_contract`；CLI 的 `generation_blockers` 给出具体路径。

## 顶层字段

| 键 | 含义 | 白皮书用途 |
|---|---|---|
| `schema_version` | Seed JSON 格式版本 | 让后续解析器知道当前结构版本 |
| `seed_id` | 当前种子的稳定标识 | 运行、审计和复现实验时区分不同领域种子 |
| `family` | 领域族，例如 `synthetic_reporting` | 帮助选择场景名称、默认文档媒介和能力组合 |
| `title` | 面向人的种子标题 | 白皮书的场景标题和摘要来源 |
| `description` | 真实工作区所表达的领域概览 | 约束白皮书的主题边界 |
| `sources` | 原始文件清单、哈希、角色和来源元数据 | 生成上下文保留来源 ID、使用权限、资料性质、日期范围及允许用途；路径、哈希和敏感性标记保留为审计信息 |
| `task` | 工作目标、业务指令和禁止事项 | 定义世界要模拟的任务，以及不能凭空推断的内容 |
| `document_map` | 文件或文件片段对应的主题、章节、表格和证据位置 | 保存为审计信息；供生成使用的文档体裁和业务要求应明确写入示例、蓝图或机制字段 |
| `exemplars` | 经过筛选或脱敏的示例文档 | 提供文风、文档体裁、字段承载方式和时间节奏 |
| `mechanisms` | 领域中真正需要维护的业务机制 | 对应评测能力，指导陷阱和变化设计 |
| `blueprint_requirements` | 实体、字段、关系、事件、因果和时间要求 | 是生成结构化世界的核心蓝图 |
| `generation_contract` | 必须满足、必须区分、禁止创造和敏感内容规则 | 作为白皮书和世界生成的硬约束/软约束 |
| `review` | 抽取覆盖率、冲突、缺失和人工决策记录 | 防止把不确定推断误当成领域事实 |

## `sources`

每个来源建议包含：

```json
{
  "id": "doc_legal_001",
  "path": "法律/附件1.pdf",
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "role": "corpus",
  "source_kind": "normative",
  "title": "...",
  "date_scope": {"published": "2026-06-24"},
  "allowed_use": ["ontology", "mechanism", "exemplar_style"],
  "sensitivity": "public"
}
```

- `id`：其他所有 `source_refs` 使用的稳定 ID。
- `path`：相对于真实工作区的路径。
- `sha256`：确保后续仍然引用同一个文件。
- `role`：固定取值 `corpus`、`builder_only`、`evaluator_only`，用于控制生成与审计的输入权限。
- `source_kind`：例如 `normative`、`factual_record`、`analytical`、`task`、`structured_dataset`、`example`，记录资料性质。
- `date_scope`：区分发布日期、报告期、观测期和事件发生期。
- `allowed_use`：说明该文件可以用于什么，避免把评估材料误当生成材料。
- `sensitivity`：决定是否需要脱敏或仅保留结构。

## `task`

建议包含：

```json
{
  "objective": "维护逐步到达的复核记录",
  "instructions": [
    "缺乏证据时保留待核状态",
    "不能把缺失解释为零"
  ],
  "forbidden_inferences": [
    "不能由新闻报道推导真实处罚结果"
  ]
}
```

- `objective`：白皮书要模拟的实际工作目标。
- `instructions`：领域操作规则。
- `forbidden_inferences`：生成世界时不能跨越的推理边界。

## `document_map`

它把文件和具体主题连接起来：

```json
{
  "source_id": "doc_legal_001",
  "segments": [
    {
      "locator": "page:2-4",
      "topic": "线上公示信息与线下经营信息不一致",
      "contributes": ["entity", "relation", "event", "mechanism"],
      "evidence_strength": "high"
    }
  ]
}
```

`document_map` 顶层使用数组，上面展示其中一个元素。它保留“哪个文件的哪一段涉及什么”的审计记录。只有明确写入生成字段、带有相应来源定位的要求会进入生成上下文；文档地图中的额外文字不会整体传给作者。

## `exemplars`

示例文档用于教白皮书和后续渲染器“信息应该以什么形式出现”：

```json
{
  "title": "合成复核记录",
  "content": "脱敏后的短文本",
  "doc_type": "复核记录",
  "date": "2026-01-15",
  "source_refs": [],
  "mechanism_refs": ["evidence_sufficiency"],
  "origin": "synthetic"
}
```

合成示例采用 `origin: "synthetic"`、空 `source_refs`、非空 `mechanism_refs`，通过机制记录来源依据。来源摘录采用 `origin: "source"`，引用 `corpus` 来源并包含准确定位。示例主要提供：

- 文档体裁；
- 字段如何自然出现；
- 实体和关系如何共同出现；
- 事件如何分散在不同时间的文档中；
- 证据充分和证据不足如何表达。

## `mechanisms`

机制是 seed 的“业务不变量”。建议结构为：

```json
{
  "id": "evidence_sufficiency",
  "description": "材料到达并复核后，才能改变证据充分性",
  "required_entity_types": ["merchant", "review_ticket", "evidence_record"],
  "required_relation_types": ["ticket_for_merchant", "evidence_for_ticket"],
  "required_event_types": ["evidence_received", "review_updated"],
  "required_causal_rules": ["evidence_to_review"],
  "failure_modes": [
    "把待补证当成已证实",
    "跳过材料直接给出结论"
  ],
  "capability_hooks": ["L2_relational", "L3_process", "L6_refusal", "L8_transition"],
  "source_refs": [{"source_id": "doc_legal_001", "locator": "page:2-4"}]
}
```

- `required_*`：生成世界必须具备的结构。
- `failure_modes`：受测系统可能犯的错误，供白皮书设计测试陷阱。
- `capability_hooks`：建议激活的 L1–L10 能力线，实际启用仍由白皮书规划和可用结构共同决定。

## `blueprint_requirements`

### `entity_types`

描述世界中有哪些不同类型的实体：

```json
{
  "id": "review_ticket",
  "noun": "复核工单",
  "count": 3,
  "fields": [
    {
      "name": "证据充分性",
      "kind": "status",
      "allowed_values": ["待补证", "充分", "不足"],
      "evolving": true,
      "source_refs": [{"source_id": "doc_legal_001", "locator": "page:2-4"}]
    }
  ]
}
```

白皮书使用它决定实体类型、字段白名单、值域和时间演化方式。

### `relation_types`

描述实体之间的类型化连接：

```json
{
  "id": "evidence_for_ticket",
  "from_type": "evidence_record",
  "to_type": "review_ticket",
  "field": "关联工单",
  "temporal": false,
  "min_count": 1,
  "source_refs": [{"source_id": "doc_legal_001", "locator": "page:2-4"}]
}
```

白皮书使用它规划多跳路径、关系文档和桥接实体，避免只生成孤立字段。

### `event_types`

描述会改变世界状态的事件：

```json
{
  "id": "review_updated",
  "label": "复核结论更新",
  "roles": {
    "ticket": "review_ticket",
    "evidence": "evidence_record"
  },
  "effect_fields": [
    {"role": "ticket", "field": "证据充分性"},
    {"role": "ticket", "field": "复查结论"}
  ],
  "min_count": 1,
  "source_refs": [{"source_id": "doc_legal_001", "locator": "page:2-4"}]
}
```

白皮书使用它安排事件分布、变化点和跨文档顺序。

### `causal_rules`

描述事件之间的合法先后关系：

```json
{
  "id": "evidence_to_review",
  "trigger_event": "evidence_received",
  "effect_event": "review_updated",
  "delay_sessions": 1,
  "shared_roles": ["ticket", "evidence"],
  "source_refs": [{"source_id": "doc_legal_001", "locator": "page:2-4"}]
}
```

白皮书使用它生成“先发生什么、后发生什么、哪些跳跃是非法的”的时间线。

示例中的 `delay_sessions: 1` 表达合成调度安排。真实来源如果只说明先后顺序，应在 `generation_contract.synthetic_designs` 中记录具体间隔的设计依据，避免将人为选定的节奏表达为来源规定。

### `state_machines`

如果领域存在有序状态，使用：

```json
{
  "id": "ticket_stage",
  "entity_type": "review_ticket",
  "field": "处置阶段",
  "states": ["资料核验", "待更新公示", "关闭"],
  "allowed_transitions": [
    ["资料核验", "待更新公示"],
    ["待更新公示", "关闭"]
  ],
  "source_refs": [{"source_id": "doc_legal_001", "locator": "page:2-4"}]
}
```

### `metric_definitions`

用于金融、体育、零售等数据密集型领域：

```json
{
  "name": "营业收入",
  "kind": "numeric",
  "unit": "元",
  "period_basis": "季度",
  "definition": "报告期内营业收入",
  "comparators": ["上年同期", "上期"],
  "source_refs": [{"source_id": "doc_report_001", "locator": "sheet:指标定义!A1:D8"}]
}
```

它防止把报告期、发布日期、同比期间、单位和指标口径混在一起。

`state_machines` 与 `metric_definitions` 属于 `blueprint_requirements` 的扩展数组。保留完整跳转条件、比较口径与适用条件，不把它们压缩成状态名或单位列表。存在同名字段时，为定义提供 `entity_type` 以明确归属。

### `evidence_channels` 与 `temporal_model`

- `evidence_channels`：例如风险工单、复核记录、公告、年报、检测表、鉴定意见书。
- `temporal_model`：最少 session 数、时间粒度、报告期粒度、事件间隔和是否允许跨期回看。

## `generation_contract`

建议分为：

```json
{
  "must_include": [
    "至少两条跨实体关系链",
    "至少一条证据不足路径"
  ],
  "must_distinguish": [
    "缺失与零值",
    "发布日期与数据日期"
  ],
  "must_not_infer": [
    "不能把行业评论当成公司事实"
  ],
  "sensitive_fields": [
    "个人身份信息",
    "病历细节"
  ],
  "syntheticization": {
    "names": "replace",
    "case_numbers": "replace",
    "numeric_values": "preserve_shape_randomize",
    "dates": "shift_or_generate"
  },
  "active_lines": ["L1_timeline", "L2_relational", "L5_conflict"],
  "traps": [
    "相邻文档使用不同日期口径",
    "同一实体存在多个来源版本"
  ],
  "conditional_requirements": [],
  "unresolved": [],
  "synthetic_designs": ["示例中的一期间隔为合成调度选择，来源仅要求先补证再复核"]
}
```

它是从 seed 到白皮书的“控制面板”：哪些结构必须出现，哪些概念必须拆开，哪些内容不能由生成器擅自补全。

声明可保留 `text`、`status`、`source_refs`、`conditions`，具体允许字段见共享格式定义。需要来源支持的声明使用 `source_refs`。详细抽取推理和审查过程留在审计记录；尚未确定的业务边界可以放入 `generation_contract.unresolved`，让作者保留不确定性，禁止将其提升为事实或义务。若未决事项阻碍核心任务，在 `review.blocking_issues` 中明确列出。

## `review`

建议记录：

```json
{
  "coverage": {
    "sources_reviewed": 8,
    "sources_used": 8,
    "claims_extracted": 42
  },
  "unresolved_conflicts": [],
  "missing_inputs": ["原始 XLS 的脚注未能解析"],
  "human_decisions": [
    "将新闻报道标为 analytical，不作为规范性来源"
  ],
  "blocking_issues": [],
  "quality_status": "draft"
}
```

`review` 用于留存审查信息。生成提示词不包含整份审查原文；生成入口读取显式阻塞项，作者使用经确认的生成字段。详细原子事实和逐文件审查记录也可以保存在单独的 `<workspace-name>_review.json`，并在 `review` 中记录对应文件名和哈希。

## 离线验收和原流程入口

```text
python tools/validate_seed_packs.py seeds/<workspace-name>_v2.json --verify-sources --source-root <original-workspace>
python tools/validate_seed_packs.py seeds/<workspace-name>_v2.json --require-generation-ready
python -m pipeline.factory --seed-pack seeds/<workspace-name>_v2.json --to whitepaper
```

前两条只做本地检查。最后一条进入原生成流程并可能调用付费模型，只在用户授权生成时执行。`passed` 表示结构及所选原件核验通过，`generation_ready` 同时执行原生成入口的准入检查，包括显式阻塞项和未决／有条件的可执行结构要求；两者都没有进行业务语义或 benchmark 质量验收。

仓库内合成示例可直接进行以下离线检查：

```text
python tools/validate_seed_packs.py skills/realfiles-to-seedjson/examples/seed_v2_example.json --verify-sources --source-root skills/realfiles-to-seedjson/examples/source_workspace --require-generation-ready
```

生成时冻结源 seed，审计材料与允许传入模型的内容分别处理。评价题目所需的业务规则必须进入公开语料或答题协议；最终公开审阅与评分仅使用受测者可见材料。

## 对白皮书的总体影响

seed JSON 不直接规定每个实体的具体名称和每一天的具体数值。它规定的是世界的“语法”：

```text
seed 的描述与任务
    → 白皮书的场景目标
实体类型与字段
    → 世界 schema
关系类型
    → 多跳结构和桥实体
事件与因果规则
    → 时间线、状态变化和合法顺序
机制与失败模式
    → 激活的能力线和测试陷阱
示例文档与证据渠道
    → 语料体裁、文风和证据分布
生成契约与审查结果
    → 硬约束、拒答边界和脱敏规则
```

因此，一个好的 seed 应该让白皮书能够回答：

1. 这个世界里有哪些实体和字段？
2. 哪些实体之间存在可追踪的关系？
3. 哪些事件会改变哪些字段？
4. 哪些事件必须先后发生？
5. 哪些值、状态和日期不能混用？
6. 受测系统最容易在哪些地方犯错？
7. 生成器不能声称或推断什么？
