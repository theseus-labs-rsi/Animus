"""
pipeline.prompts —— 所有 LLM prompt 的【单一注册表 + 渲染层】。

工程目标:把散落在 central_office.py / run_factory_v2.py 里的 prompt 字符串常量 + inline f-string 收口到一处,
代码侧只调 render("<stage>.<role>", **vars),不再夹带 prompt 文本。

约定:
- 占位用 $var(string.Template)。★prompt 里大量的 JSON `{}` 保持【字面】,不与占位冲突(避开 .format 的转义地狱)。
- 模板里【只放文本】;表达式(n_sessions-1 / json.dumps() / sorted(blocked)[:30] 等)在【调用侧】算好再传进来。
- 键名 = "<阶段>.<角色>",如 world.system / corpus.user / phrase.system。

本文件先收编【run_factory_v2 渲染侧】prompts;议会(central_office)prompts 待其稳定后并入。
"""
from string import Template

PROMPTS: dict[str, str] = {

    # ── §5 世界生成(system;域参数化:$noun $fdesc $stopped）──────────────
    "world.system": """你是 memory benchmark 的 ground-truth 世界设计师。$world_scope\
代码会据此机械算标准答案,你只填表——★严禁写问题/答案,严禁"最新/当前/截至…仍为"这类结论性措辞。

【每个 $noun = 一个 entity】★核心:name 是这个【$noun】个体的【正式专名】,且必须【与「$noun」这个类别相称】——\
名称形式必须对应【$noun】：组织用组织名，文件/报告/来源记录用文档标题，事件用事件名称，人物用人名。只命名当前类型的对象；不得用对象的出处、作者、部门或数值指标代替对象本身。
★字段【只能从下面这份给定清单里选】,逐字照抄字段名,严禁新增/改名/拆分同义字段(下游 gold 只认这份清单,擅自加的字段会被丢弃且制造近义串味坏题):$fdesc。
$numeric_policy
- person/status/category 字段:给 trajectory(随时间换),或 stable 给单一 value；但 user 若标为【domain event 驱动】，这里只能给可选初态，后续变化留给 event effect
$coverage_policy

【硬约束】1.全新虚构值(防泄漏);2.session 用 0..N-1 整数,不写日期;3.evolving≥2 个不同值,★字段就用给定清单里的全部【非结构驱动】字段(不另加、不少给；关系/event 驱动字段遵从 user 的单独说明);4.$identity_policy

若给定字段清单是“无内在字段”,就输出空 fields,不要发明字段；关系字段会由结构编译器另行写入。

【严格 JSON,name 是专名而非字段,type 必须逐字为 type id】{"entities":[{"name":"<一个真实$noun的专名>","type":"$type_id","fields":{"<字段名>":{"type":"evolving","value_type":"...","trajectory":[{"session":0,"value":"..."}]},"<稳定字段>":{"type":"stable","value":"..."}}}]}""",

    # ── 世界生成 user($noun $want $smax $extra;smax = n_sessions-1;extra=白皮书 change_density/traps 钩子)──
    "world.user": """设计 $want 个【$noun】(type id=$type_id,名字互不相同),session 用 0..$smax；$trajectory_request。严格 JSON。$extra""",

    # ── 世界骨架实例化：只连已生成实体，不再发明实体/字段/关系/事件类型 ──
    "world.structure": """你是世界蓝图实例化器。给定【已经生成的 typed entities】和【机器契约 blueprint】，只实例化契约声明的关系与领域事件。
【硬约束】
1. 只能引用给定实体专名；relation 的 from/to 类型必须匹配声明，且每种至少满足 min_count（这是最低数量，不是恰好数量或上限）。在给定实体和时间范围内，只增加任务与业务过程确有需要的有限实例，不必为每个实体铺满，不为题型或难度强造关系变化。每个 relation id 唯一，禁止重复边；temporal=false 的静态关系 session 必须为 0。代码会根据 relation.field 在哪一端声明，将另一端专名写入该 owner 的 Timeline。对同一 owner.field，按 session 递增时必须是真实引用变化，禁止连续重复同一目标凑数量。
2. event 的 participants 必须逐角色匹配声明类型，同一事件的不同角色必须由不同实体承担；每种至少满足 min_count，因果规则和任务机制需要的事件可以超过最低数量，但不得为凑数量或出难题制造无业务依据的事件；session 为 0..$smax。event id 唯一，且同一 type/session/完整 participants 只能出现一次，禁止只换 id 重复计数。
3. 每个 event 的 effects 必须逐项且恰好一次覆盖该类型全部 effect_fields，不得漏项、重复或增加；输出时 entity 必须等于 participants 中该 role 对应实体，并给 set 值。全局每个 (entity,field,session) effect slot 只能被一个事件使用；set 必须相对 initial_state 与此前事件造成真实变化。多个事件写同一状态字段时，分配到递增 session，并沿 fields.states 合法推进；numeric 遵守 monotonic 且数值不同。禁止冲突、同 session 双写或空操作。
4. causal_rules 声明 A→B 时，至少造一对真实 A/B 事件；B 写 caused_by=A 的事件 id，session 差严格等于 delay_sessions。
5. typed entities 给出本轮已生成的 intrinsic_fields（稳定值或完整轨迹），增量已有实体给出 canonical_fields（已编译的完整操作历史）；它们是安排关系、参与者、时点和效果的事实背景，不得重写。目录也可能给出 event-owned 字段的 initial_state；effect 必须写成不同于其当时状态的真实变化，不能空操作。已有 canonical 关系与事件保持不变；增量只补必要的新实例，最低数量按已有与新增合计。
6. 严禁输出 cascades；代码会从验证过的事件因果机械编译。严禁新增类型、字段或实体。
【严格 JSON】{"relations":[{"id":"rel-1","type":"<relation id>","from":"<实体>","to":"<实体>","session":0}],"events":[{"id":"evt-1","type":"<event id>","session":1,"participants":{"<role>":"<实体>"},"effects":[{"entity":"<该 role 实体>","field":"<声明字段>","set":"<新值>"}],"caused_by":"<可省,父事件id>"}]}""",

    "world.structure_user": """【时间制度】unit=$time_unit,cadence=$cadence,session=0..$smax
【session 对应实际日期】$session_dates
【world_blueprint】$blueprint
【typed entities】$entities
【已有 canonical 结构；空数组表示没有已有实例】$existing_canonical
结合完整字段事实和任务背景实例化全部最低数量约束；不要更改已给定事实，不为题型或难度增造实例。严格 JSON。""",

    # ── game-only 轻量 Story Ledger；canonical world 冻结后独立生成 ──
    "world.story": """你是游戏剧情编排器。给你的世界实体和事件已经通过机器契约并被冻结；不得修改、补写或重新生成它们。
你只需把全部 canonical event 编排成一条易于渲染的剧情主线：
1. premise、goal、stakes 各写一个非空短句；protagonist_ref 必须是唯一 primary 实例。goal 必须能被给定 canonical events 在最后一幕真正完成，不能许诺事件表里不存在的“最终挑战”。
2. 每个 scene 至少引用一个 event；每个 canonical event ID 必须且只能出现一次。scene.session 必须与其引用的所有 event session 相同。
3. 同一 session 有多幕时用 order=0,1,... 排列；所有 session/order 都必须写 JSON 整数，不能写字符串。
4. dramatic_function 只允许 setup、inciting_incident、rising_action、reversal、crisis、climax、resolution；不得倒退，三幕以上时必须从 setup/inciting_incident/rising_action 起步，并以 resolution 收束。
5. scene 只承担排序和事件引用，不写自由摘要；事件内容由代码按 event_refs 从 CANON 读取。
严格 JSON，只输出：{"premise":"...","goal":"...","stakes":"...","protagonist_ref":"...","scenes":[{"scene_id":"scene-...","session":0,"order":0,"event_refs":["evt-..."],"dramatic_function":"setup"}]}""",

    "world.story_user": """【world_blueprint】$blueprint
【typed entities】$entities
【canonical events】$events$hint
编排全部 canonical events，严格 JSON。""",

    # ── game 叙事共用的只读 supportedness 闸；不改 canon，只指出编造 ──
    "narrative.review": """你是只读的叙事事实审查员。判断候选文本中每个具体的身份、状态、行动、持有关系、物品来源、阵营归属、生死、因果、结果和时间断言，是否被 CANON 明确支持。
允许文风化连接、情绪和明确标为目标/风险/假设的语句；明确以“若失败”表述的 stakes 无需在 CANON 中有反事实事件，只要不与 CANON 冲突。不得把可能性写成已发生事实，不得为事件添加 CANON 没有的执行者、掉落物、复活、立场变化或后果。
若 CANON 给出 period_label/document_date，它们就是本篇允许使用的章/期标签与文档日期；“日志记录/档案登记/通报提及”等纯文档载体动词也是允许的写作支架，不应被误判成新的领域事件。它们不能借机新增谁执行了游戏行动或产生了什么结果。
若审查对象是 Story Ledger，还要检查 premise/goal 是否与整条事件链一致、goal 是否确实在最后一幕完成；未完成的许诺也要报告，但不要仅因 stakes 是反事实风险而报告。
若审查对象是语料，还要检查每个 CANON domain_event 的动作、全部参与者和全部 effect 是否在同一篇文档里被明确叙述；只罗列状态、分散在多篇或漏掉事件也要报告。
只报候选中不受支持或未闭合的具体问题，每条一句；全部通过则返回空数组。
严格 JSON:{"unsupported_claims":["..."]}""",

    "narrative.review_user": """【审查对象】$scope
【CANON（唯一事实边界）】$canon
【候选】$candidate
只输出严格 JSON:{"unsupported_claims":[]}""",

    # ── §W.3 世界修复轮(system;定向重生成有缺陷字段;$noun $smax)──────────────
    "world.repair": """你是 ground-truth 世界设计师,在做【定向修复】。给你一个【$noun】的若干【有缺陷的字段】,只重写这些字段的取值轨迹来消除缺陷——别动其它字段、别改实体名、别新增字段。
【缺陷与修法】monotonic=数值轨迹单调 → 让峰【或】谷落在【非首非尾】的中间某周(其余可起伏);fake_evolving=只有 1 个值 → 给【≥2 个不同值】的演化轨迹;illegal_transition=状态倒流/出界 → 只用缺陷里给出的【声明状态表】取值,且按表序【单向推进】(可跳级、不可回头);monotonic_violation=该字段语义只增(或只减)却逆向了 → 重写成单向【不减/不增】(可个别周持平、整体要演化)的轨迹(如累计量逐周递增);out_of_range=值出界 → 重写到给定值域内。数值写纯阿拉伯数字、不加千分位逗号。
【硬约束】session 用 0..$smax 整数;全新虚构值;只输出被点名的这些字段。
【严格 JSON】{"fields":{"<字段名>":{"type":"evolving","trajectory":[{"session":0,"value":"..."}]}}}""",

    # ── 世界修复 user($noun $ent $defects $smax)─────────────────────────────
    "world.repair_user": """【$noun:$ent】以下字段有缺陷,只重写这几个(session 0..$smax):
$defects
严格 JSON。""",

    # ── §7 信号渲染(system;$noun $genres $stopped $genre0)──────────────────
    "corpus.system": """你是【$noun】领域的语料合成专家。场景里有多类对象：$type_legend。给你某一$time_unit各 typed entity 的当期字段值与领域事件,合成 1-3 篇异质文档(体裁:$genres),像【真实的该体裁文档】那样把这些值与事件【自然叙述】进去。
【白皮书写作规格】$style_spec
写作规格控制语气、格式、术语和信息显隐。其中篇幅是风格目标，不是可以牺牲事实的硬上限；事实较多时拆成多篇，始终以完整、忠实承载为先。
★每条 fact 的 entity_type 决定它是什么对象；严禁把 Boss/装备/项目/预算等一律称作 primary 类型。若给了 domain_events，文档应以“发生了什么”组织叙事，并忠实承载其 participants/effects；每个 event 的动作语义、全部 participant 专名、每条 effect 的实体/字段名/set 值必须出现在同一篇文档里。label 可自然改写，不要求逐字复述，但不能退回字段清单。
【就近·硬约束(是就近、不是句式)】每条事实里,【该实体的专名】与【它的值】要落在【同一句或紧邻一句】(下游有盲读者逐条校验"据本文,该实体的该字段是多少")。但这只要求【挨得近、能被唯一读出】、不规定句式——用真实文档的行文把值带出来:★绝不要写成「<实体>本期<字段>为<值>」这种字段表口吻,也不要逐字段平铺罗列。示例:写「复盘会上,星海广场项目的风险评分已抬到 80,主办人陈明据此提示团队收紧排期」,而非「星海广场本期风险评分为80。本期主办人为陈明。」
【防剧透·硬约束】1.只写本期快照值,严禁"当前/现在/最新/目前/一直/维持/累计/现任/仍为"等全局口径词(字段名本身含这些字的照常写);2.严禁回顾历史值/叙述"由X变Y";3.含本期日期锚点;4.某字段本期 stopped 就自然写明「自本期起$stopped」、不写其过去数值;5.绝不编造未给定的字段/值;6.★数值/专名的【值】逐字保留(★给的是 0.78 就写 0.78,绝不换算成 78%/78 分;给的是 320万 就写 320万,不去单位),但承载它的句子自由发挥;7.同一实体的关系/负责人写清楚、别让一个实体冒出多个互相矛盾的负责人(否则盲读者读不出唯一答案)。
【★元话术禁令】正文只写文档内容本身;★绝不在文中复述或声明你遵守了哪些约束(如"未使用全局口径词""无历史回顾""所有字段均就近""本期快照"之类说明性元话术,一律不得出现在正文)。
【严格 JSON】{"docs":[{"type":"$genre0","content":"...","fact_refs":["实体.字段"]}]}""",

    # ── 信号 user($s $date $facts $hint)────────────────────────────────────
    "corpus.quality_system": """你是【$noun】领域的语料作者。对象类型：$type_legend。体裁可选：$genres。
【白皮书写作规格】$style_spec
把给定本期需要披露的 facts 和 events 自然写进 1-3 篇文档。篇幅以完整、忠实承载为先。
清楚表达主体、字段含义、时点、值的单位量纲，以及事件参与者的角色、动作和全部结果。
允许自然同义表述和清楚指代，不必照抄字段名或枚举值，也不以出现某些关键词代替事实。
正文以本期记录为中心并含给定日期；可以准确区分已知历史与本期变化，不能预告未来真值。
停用字段应按其给定的原生效时点自然说明$stopped，不把较晚的披露日期改写成停用生效日期，也不把旧值继续写作有效值。
数值与关系含义不得改变，不能猜补未给定的主体、单位、来源身份或业务动作。
计划、传闻、否定、角色误写和载体登记须保留其语气，不把它们升级成实际业务状态。
完整上下文只用于核实含义，不要求把全世界历史重复到正文。本组要求不能依赖标题或隐藏元数据。
正文只写文档本身，不复述生成约束或答案提示。
严格 JSON：{"docs":[{"type":"$genre0","content":"自然正文"}]}。""",

    "corpus.quality_user": """【第 $s $time_unit / $date】本组需要公开表达的事实（各事实保留自身时点）：$facts
【本组需要公开表达的事件（保留原发生时点）】$events$story_context$hint
这些是待表达的冻结事实，不是固定句式。用自然文档准确传达，严格 JSON。""",

    "discriminate.quality_system": """你是只读给定文档的盲读者，不知道隐藏世界或期望答案。
逐项说明正文对该实体字段表达的值或状态；字段停用时说明正文的停用含义。
允许用自然语言说明读到的含义，保留主体、时间、单位、否定、条件和不确定性。
不要从外部常识或缺失前提补答案。正文不足、指代不清或说法不一致时如实说明。
每个输入 key 必须且只能返回一次。严格 JSON：{"answers":[{"key":"q0","answer":"据正文读到的含义或不确定原因"}]}。""",

    "discriminate.quality_user": """【文档】
$docs
【查询】$queries
只依据这些文档逐项回答，不存在额外答案。严格 JSON {"answers":[{"key":"q0","answer":"..."}]}。""",

    "corpus.user": """【第 $s $time_unit / $date】各 typed entity 当期字段值(只写这些、只写本期):$facts
【本期 domain_events】$events$story_context$hint
严格 JSON。""",

    "corpus.public_rules.system": """你是本场景原语料的作者。根据已冻结的类型/字段阶段定义，写自然的流程说明或操作说明，供阅读本场景文档的人使用。
【白皮书写作规格】$style_spec
正文完整说明每一项的适用对象类型、字段及全部阶段的先后顺序，可以分成数篇自然文档。正文使用给定文档日期。
这是阶段顺序，不是具体实体的状态记录：不写任何实体当前处于哪里、下一步答案或未来会发生什么；不创造额外生效事件、签批人和生效日期。
实际记录可跳级、不可倒退；不能把序列中紧邻的后一阶段写成实际每次变更必须到达的唯一阶段。
不得引用题号、答案、私有来源路径或规则ID等生成元数据。不能省略规则而依赖读者常识。
只返回 JSON:{"docs":[{"type":"流程说明","title":"自然标题","content":"完整正文"}]}。""",

    "corpus.public_rules.user": """【文档日期】$date
【冻结的公共阶段定义】$rules
$hint
只写这些类型级规则，不增加具体实体事实。严格 JSON。""",

    # ── 渲染链·盲判别器(Blinded Discriminator;只读渲染文档、对世界一无所知)──────
    #   死钉③:user 只喂 $docs(渲染正文)+ 要问的 ($entity,$field);★绝不喂 value/gt/fact_refs。
    "discriminate.system": """你是一个【只读下列文档的盲读者】,对文档之外的世界一无所知,没有任何先验常识或背景知识。
你的任务:只依据【给定文档】逐项回答每个实体字段【字面写的是什么值】。
【铁律】
1. ★只回文档里【逐字写出的那个值本身】(含单位、百分号、量纲原样照抄):文档写"78%"就回"78%"、写"0.78"就回"0.78"、写"320万"就回"320万",绝不换算、绝不归一、绝不补单位、绝不去单位。
2. 文档里【根本读不出】这个字段的值,或【多处说法不一致】,或【指代有歧义】(分不清是哪个实体的) → 回"不确定"。
3. ★绝不猜测、绝不用任何外部常识补全、绝不编造一个文档里没有的值。
4. 每个输入 key 必须且只能返回一次，不得漏项、合并或改写 key。
【输出严格 JSON】{"answers":[{"key":"q0","answer":"<逐字照抄的那个值,或'不确定'>"}]}""",

    # ── 盲判别器 user($docs $queries)─────────────────────────────────────────
    "discriminate.user": """【文档】
$docs

【问题列表】$queries
逐项回答 entity 的 field；只回该值本身(逐字照抄,含单位/百分号/量纲),文档里读不出或有歧义或多处不一致就回"不确定"。
严格 JSON {"answers":[{"key":"q0","answer":"..."}]}。""",

    # ── §7 草堆渲染(system;$noun)─────────────────────────────────────────
    "filler.system": """你为记忆评测生成一篇【与目标场景同一领域、同一叙事世界】的背景干扰文档。草堆必须像这个世界自然产生的旁支记录,不能突然切换成另一个行业或时代。
【领域画像】主体类别:$noun；优先体裁:$genres；外围主题池:$filler_topics；冻结世界语境:$world_context
【硬约束】
1. ★绝不碰任何被追踪的 $noun 专名或人物专名(连名字都不出现),只使用全新虚构的外围专名与无关旁支事件,不得承载可用于回答 benchmark 问题的事实。允许使用“状态/工具/负责人”等同领域通用词，但不得把它们与被追踪专名组合成可回答事实。
2. 文体、术语、时间制度必须服从上述领域画像。例如 RPG 应写世界内告示、传闻、游记、货单等,不得出现公司员工、OA、办公区、食堂培训等现代办公内容；只有目标场景本来就是企业办公时才允许办公题材。
3. 题材与体裁自然像真实文档；正文600-1000字,带本期日期。
只输出文档正文，不要 JSON、代码块、标题标签或解释。""",

    # ── 草堆 user($s $date)─────────────────────────────────────────────────
    # 受保护专名只留在代码端验收，绝不放进提示词。把禁词逐项展示给模型会反而
    # 提高复述概率；system 已声明只用全新外围专名，输出再由代码逐字 fail-closed。
    "filler.user": """【第 $s $time_unit / $date；本期草堆槽位 $slot】从外围主题池中选择适合该槽位的一类，写一篇与主线无关的外围干扰文档。不同槽位应改变体裁、外围主体和事项，避免复述同一模板。
只输出正文。""",

    # ── §7 敏感注入渲染(L10;★确定性模板,绕开 LLM 那一跳 → 保证 X 逐字就近落地,G3 反退化L6）─────
    #   不走 tracer.chat_json:代码直接 Template.substitute 出成品文档(user 供出 X + assistant 已记录),
    #   使敏感值 X 在语料里【逐字 + 就近实体】出现(与 L6 藏诱饵相反),同时 X 是 forbidden 集本体。
    "sensitive.template": """【$date · 记忆写入】
用户($entity):我的$field_label是 $value,请存进记忆备用。
助手:已记录。$entity 的$field_label我已安全存档,后续需要时可为你调取。""",

    # ── §7 条件归纳实例渲染(L9;★确定性模板,绕开 LLM → 保证只渲【单条情境→动作】、禁一般化规则句)─────
    #   代码直接 Template.substitute:每条执行实例 = 某主体在某 trigger 值下【实际采取的一个动作】(用 surface 表面串,canon 层)。
    #   ★铁律:只叙述【这一条】情境与处置,绝不出现"若/当…则""凡…一律""阈值""规则如下"等一般化措辞
    #   (那会把待归纳的规则直接写进语料 → 退化成检索题)。
    "rule.template": """【$date · 处置记录】
$inst_id 本次登记的「$trigger_field」为 $x$unit。值班据此对该 $inst_id 作出处置:$surface_action。本条记录归档。""",

    # ── ④ 出题(system 常量)────────────────────────────────────────────────
    "phrase.system": """你是出题人。把给定的【提问意图】写成一句【自然、口语、像真人问】的问题。
铁律:① 题面绝不出现【答案】或【须隐藏的中间值】;② 不添加意图之外的设定/数值;③ 只输出问题本身,一句话;
④ ★【忠实保留意图的疑问类型/答案口径】:意图问"是多少"(数值)就别改写成"是谁";问"是谁"(人)别改成"是多少";问"哪一周"别改成问值——只改措辞,绝不改答案类型(否则问法与标准答案对不上)。
⑤ ★【保留主语专名】:意图里【被问的那个实体专名】(部门/案件/患者/人名…)必须在题面里【原样出现】,绝不可用"他/她/它/该案/该部门/其"等代词替代或省略——否则题面失去指代、无法作答(代码会核验,丢了就退回原意图)。
只输出 JSON:{"question":"..."}""",

    # ── 出题 user($intent $hide)────────────────────────────────────────────
    "phrase.user": """【提问意图】$intent
【题面绝不可出现的词】$hide
写成一句自然问题,严格 JSON。""",

    # ── §3 中央议会:6 视角 system + 批判 system(从 central_office 收编)──────
    "council.observe": """你只做【据实抽取】,绝不推断、不脑补。从 few-shot 文档里,只抽取【字面明确出现】的东西。看不出来的可选键直接省略，禁止输出“可省/无/未知/不适用”等占位值。
★ main_entity_noun = 被【逐周跟踪、各字段所描述】的那个【主体】类别(如 部门/SKU/患者/案件)——它是"周报在讲谁"。注意:person 类字段(负责人/汇报对象)是主体的【关系对象】,不是主体本身;别把它当 main_entity。
★numeric 字段按字面+领域常识补三个【值约束】(只在明确时标,不确定就省,绝不硬凑):
  · unit:字面带计量单位(「320万」→"万"、「86 小时」→"小时");
  · monotonic:值【只增不减】(累计/合计/总量/里程类,如「累计计费工时」)标 "up";【只减不增】(剩余/倒计/待办类)标 "down";
  · range:有【固定值域】(评分 0–10、百分率 0–100、达成率 0–1 等)标 [下限,上限]。
只输出 JSON:{"main_entity_noun":"...","observed_media":["体裁..."],"observed_entities":["实体类型..."],"observed_fields":[{"name":"..","kind":"numeric|person|status|category|text"}]}。unit/monotonic/range/stopped_phrase_seen 均为可选键，只在明确时添加；range 若有必须为两个 JSON 数字。""",

    "council.skeptic": """few-shot 只是这个场景的【针孔样本】,系统性缺失。你以【本领域资深从业者】身份,阐发它【没展示、但此场景真实存在】的东西。
★缰绳(必须遵守):(a) 每一项都得是"领域专家会点头承认'此场景确实有它'"的;(b) 给 confidence(high/med/low)+ 一句 why;(c) 宁缺毋滥,绝不编造猎奇。
只输出 JSON:{"latent_media":[{"form":"..","confidence":"..","why":".."}],"latent_entities":[{"type":"..","confidence":"..","why":".."}],"latent_relations":[{"type":"..","from":"..","to":"..","confidence":"..","why":".."}]}""",

    # ★map 内联宪法 taxonomy:用 $taxonomy 占位,调用侧传 taxonomy_prose()
    "council.map": """对照下面这套【记忆挑战坐标(产线 L1–L7)】,逐条判断:本场景是否【天然支持】这条产线?若支持,用本场景的什么【具体结构】落地(哪些字段/关系/事件/偏好/矛盾),并给权重建议(0–1)与 gt 可行性。
【产线坐标】$taxonomy
★若 L4_preference 适用,额外给 `preference_axis`，但它必须引用已冻结 world_blueprint 中某个 entity type 已声明的真实重复选择字段：{entity_type,field,options}。不得新增字段、不得要求 L4 事后造时间线；自然轨迹不够形成稳定偏好就判不适用。
★生命周期状态序已经冻结在 world_blueprint.fields[].states；这里只能引用它判断能力是否适用，严禁另写一套 state_machines 覆盖世界。
只输出 JSON:{"per_line":[{"line":"L1_timeline","applicable":true,"instantiation":"本场景用..落地","gt_feasible":true,"weight_hint":0.4}],"preference_axis":{"entity_type":"...","field":"...","options":["...","..."]}}(L1–L7 每条都判一次;preference_axis 可省)""",

    "council.medium": """发散本场景【所有可能的文档/记录形式】。先穷尽【常见】形式(力求大而全),再补【反常但合理】的(人类一下子想不到、但此领域确实可能存在的)——每个反常项必须给"为何此场景合理"。不是猎奇,是覆盖完整。最后给主媒介组合建议。
另外规划 benchmark 的【草堆语料】。草堆是在同一领域、同一世界里自然出现的外围记录，用全新外围专名和与主任务无关的旁支事项增加阅读总量；它不能承载被追踪实体的答案、世界真值、规则答案或未来信息。请给出每期 8–12 篇的建议、至少 4 种适合该领域的外围体裁、至少 6 类可轮换的外围主题。规模建议要考虑真实记录密度和体裁篇幅，不能靠重复同一模板灌水。
只输出 JSON:{"common_media":[".."],"unconventional_media":[{"form":"..","why_plausible":".."}],"recommended_mix":["..",".."],"haystack_plan":{"filler_documents_per_session":10,"filler_genres":[".."],"filler_topics":[".."],"why":".."}}""",

    "council.style": """从 few-shot 原文抽【风格 DNA】,供后续渲染【照着仿写】:语气、格式(连续段落?条目?表格?)、典型篇幅、术语/黑话密度、什么明说·什么默认。
只输出 JSON:{"style_spec":{"tone":"..","format":"..","length":"..","jargon":"..","stated_vs_assumed":".."},"use_fewshot_as_exemplar":true}""",

    "council.traps": """设计本场景【天然在哪坑记忆系统】,让 benchmark 有区分度:recency 偏置 / 长程依赖 / 多源矛盾 / 易混字段(语义近但不同名) …。★不要用"近重名实体"(同主干近重名如 张三/张三(数据))——它在渲染层会塌缩成歧义、制造无唯一解的坏题,已禁用。每个陷阱说明它考验哪种记忆失败,并建议该【重激活哪条产线】来制造它。
只输出 JSON:{"traps":[{"trap":"..","stresses":"..","boost_line":"L?_.."}]}""",

    "council.world": """你是【领域世界架构师】。你的任务不是列几个字段，而是先回答：这个场景的世界在骨子里由哪些不同类型的对象、关系和领域事件构成，它按什么时间制度变化？
从场景描述与 few-shot 出发，用领域常识补足针孔样本看不到但真实、必要的结构；不要按记忆题型反推世界，也不要把所有对象压成同一种 entity。
observe.observed_fields 是字面硬事实：其中每个 name 必须逐字出现在至少一个合理 entity type 上，observe 明示的 kind/unit/monotonic/range 也必须原样复制；它们可以只是外围观测字段，不能取代领域核心拓扑与事件。
★observe 的 person/category/text 是 few-shot 中的【字面显示值】，必须按原 kind 保留；它不等于 typed relation。若同一语义还需要可遍历关系，另加一个名称不同的 reference 字段承载关系，绝不能把已观测字段改成 reference，也不能复用非 reference 字段充当 relation.field。
★先逐项核对场景描述里的核心循环：凡是有独立身份、会参与关系/事件、需要被持续追踪的核心名词，都必须成为 entity type，不能降成 category/text 来省事；描述明确列出的核心关系与核心事件必须分别有 relation/event 声明。例如“阵营控制地区”必须有 faction 与 region 两端，不能偷换成“玩家控制地区”；若描述分别列出“掉落、拾取、装备”，就不能合并成一个含糊事件。

输出一个可执行 world_blueprint v1：
- entity_types：每类有稳定英文 id、中文 noun、count、唯一一个 primary=true、该类型【专属】fields。字段沿用 {name,kind,unit?,monotonic?,range?,states?}；kind 只能逐字使用 numeric/person/status/category/text/reference，整数与浮点数也统一写 numeric，禁止写 number/integer/int/float；关系字段也必须先声明，kind 用 reference。`states` 只声明不可逆、单向推进的生命周期；会因重试、重开、恢复、上下线而回到早期状态的运行状态只写 kind=status，必须省略 states。
- relation_types：{id,from_type,to_type,field,temporal,min_count}。from→to 必须能按“from 谓词 to”读成领域自然语义，绝不能因 field 在另一端就反转或替换端点；field 必须逐字声明在两个端点类型中的【恰好一端】，该端就是软外键 owner，值指向另一端。自关系约定 from/source 持有字段。v1 每个 reference 字段必须且只能绑定一种 relation，不能悬空，也不能同时被 event effect 写入。每种 relation 的 min_count>=1。一对多通常把 field 放在“多”的一侧，多对多改用关联实体。
- event_types：{id,label,roles:{角色:type_id},effect_fields:[{role,field}],min_count}。label 是文档可自然逐字使用的人类可读事件名（如“击败首领”“预算修订”）；不同角色必须由互异实体承担，因此同类型角色出现 N 次时该类型 count 至少为 N；每个事件至少一个真实状态效果且 min_count>=1。描述中分开的领域动作不得为了省 schema 被合并。
- entity 的 name 已经是稳定专名；除非 observe.observed_fields 字面要求，否则不要再把“XX名称/姓名/编号/ID”声明成 fields（只作为关系端点的类型允许 fields=[]），更不能拿它充当事件效果。event effect 必须落到会真实变化的状态、数值或类别字段；若一个核心动作尚无可写效果，应补领域自然的生命周期字段（例如物品流转状态：未掉落→已掉落→已拾取→已装备），不能用“名称 SET 成自身”伪造变化。
- temporal_model：{unit,cadence,n_sessions,step_days}。unit 可为 week/chapter/business_day/round/event 等；n_sessions>=2，step_days>=1。
- causal_rules：只有领域中无需额外主体选择、稳定必然成立的事件因果才写 {id,trigger_event,effect_event,delay_sessions}，没有就空数组；中间有人类/玩家选择时必须拆开，不能把“击败→掉落”偷换成“击败→拾取”。
- evidence_channels：这个世界里真实留下痕迹、可观察核心事件的文档/记录渠道。

★v1 不接受不可执行的 prose invariants。约束必须落到 fields.states/range/monotonic、relation_types、event_types.effect_fields 或 causal_rules 中；无法机械表达的只作为评审意见，不得伪装成已执行契约。
★可选字段不适用时直接省略（或为 JSON null），绝不拿 [0,0]、[null,null] 等占位。range 仅用于 numeric 且必须是两个不同的 JSON 数字；reference 的目标类型只由 relation_types 表达。monotonic 仅可写 up/down。只有一个 entity type 的 primary=true，其余必须 false。

★所有 id 唯一、引用闭合；跨类型同名字段若存在，kind/unit/monotonic/range 必须完全一致。至少设计 2 种实体类型、1 种关系和 1 种事件，形成该场景自己的拓扑与事件生态。不要提 L1–L10。
只输出 JSON:{"world_blueprint":{"version":1,"entity_types":[{"id":"...","noun":"...","count":3,"primary":true,"fields":[{"name":"...","kind":"status"}]}],"relation_types":[{"id":"...","from_type":"...","to_type":"...","field":"...","temporal":true,"min_count":1}],"event_types":[{"id":"...","label":"人类可读事件名","roles":{"actor":"..."},"effect_fields":[{"role":"actor","field":"..."}],"min_count":1}],"temporal_model":{"unit":"week","cadence":"weekly","n_sessions":10,"step_days":7},"causal_rules":[],"evidence_channels":["..."]}}""",

    "council.world_repair": """你是【world_blueprint schema 修理员】，不是世界架构师。候选世界的领域意图已经确定；你只能依据机械错误清单修复 JSON 契约，不能删掉实体/关系/事件来逃避校验，也不能重新发挥另一套世界。
逐项复核：
- 恰一个 primary=true；其余 false。
- 每个 relation 的 from→to 保留领域自然语义；field 必须在两个端点类型中的【恰好一端】逐字找到，编译器会把该端判为 FK owner、值写成另一端。自关系由 source 持有。不要为了字段 owner 颠倒“Boss→掉落装备”“阵营→控制地区”等自然方向。
- relation.field 必须 kind=reference，且每个 reference 字段必须且只能绑定一种 relation（禁止悬空或复用）；event effect 不得再写 relation-owned 字段。静态关系每个 owner 最多承载一个实例；时变关系也必须产生真实引用变化，不能重复同值凑 min_count。
- 显式 v1 的每种 relation/event 都必须 min_count>=1；不允许靠设为 0 逃避实例化。场景描述明确分开的核心对象、关系和事件必须保留，不得偷换端点或合并动作。
- 每个 event role/effect 引用闭合；effect 只能写该 role 类型已经声明的字段。
- 字段 kind 只能逐字使用 numeric/person/status/category/text/reference；看到 number/integer/int/float 一律改成 numeric。数值字段若带 range/monotonic，kind 必须同时为 numeric。
- “XX名称/姓名/编号/ID”等身份字段不得作为 event effect。若核心事件没有可变化字段，新增领域自然的状态/数值/类别字段；禁止用“名称 SET 成实体专名”伪造效果。
- entity.name 已经承载 canonical 专名；不在 observe 冻结清单、也不被结构引用的重复身份字段应删除。只作为关系端点的类型允许 fields=[]，不要为满足非空而补“名称”。
- range 只放 numeric，且为两个不同 JSON 数字；reference/status/category/text 不写 range。monotonic 只写 up/down。
- `states` 只用于不可逆单向生命周期；重试、重开、恢复、上下线等可循环运行状态必须删除 states，只保留 kind=status，不能为了通过校验篡改真实事件顺序。
- 跨类型同名字段约束若不同，改成领域清楚的不同字段名，并同步全部引用。
- 错误若指出“未覆盖/改写 few-shot 字段”，必须把该字段名及明确的 kind/unit/monotonic/range 逐字恢复到合理类型；不得用英文翻译或近义词替代。
- person/category/text 等已观测显示字段必须原样保留；若还要建立关系，新增不同名的 reference 字段承载，禁止改型或把非 reference 字段直接用作 relation.field。
- v1 禁止 prose invariants；候选里若出现 `invariants`，必须删除该键或置为空数组，把能表达的约束改写进 fields/relations/events/causal_rules。
只输出 {"world_blueprint":{...}} 完整 JSON。""",

    "council.world_repair_user": """【机械错误】
$errors
【待修候选】
$candidate
严格逐项修复并输出完整 JSON。""",

    "council.critic": """你是中央办公室【批判员】。审一份白皮书,挑硬伤并直接产出【修订后的完整白皮书 JSON】(不是 diff,照原 schema):
1. 域解析贴合 few-shot 吗?inferred 的有没有不合理(违反怀疑缰绳)?
2. active_lines 是否真贴合本场景天然结构、是否制造了区分度(别什么场景都只配 L1)?gt 都可行吗?
3. shared_world_spec 的实体/关系/周数撑得起激活的产线 + 目标题量吗?关系够 L2 用吗?
4. medium 选得对吗?style_spec 能指导渲染吗?
★【产线 id 铁律】active_lines 只能保留草案中已经激活的条目，严禁新增、重命名、自造名或合并；不适用线只留在 line_mapping 的审议记录中，不能以 weight=0 塞回 active_lines。你只能修改既有 active_lines 的正权重与 why：
$taxonomy
★【轴字段铁律】preference_axis 若存在，必须指向 world_blueprint 已声明且归属唯一 entity_type 的真实字段；能力线只读该自然轨迹，严禁另造/改写偏好时间线。
★【字段约束铁律】field_schema 各字段的【名字】及其上的 unit / monotonic / range 约束逐字保留,不得删、不得改、不得改名(它们是下游 gold 正确性的硬依赖;同名字段的 unit/mono/range 若被你改动代码会以草案为准还原,但改名无法自动还原、会丢约束,所以务必别改名)。
★【世界骨架铁律】world_blueprint 是领域架构师已通过机械校验的可执行契约；整块逐字保留，严禁删除、改写、扁平化或按 active_lines 反推重构。代码也会强制以草案版本覆盖。
照原 schema 输出修订版完整白皮书 JSON(保留 preference_axis / state_machines 等 domain_profile 字段,不得丢)。""",

    # ── 议会 user 模板:7 视角共用($desc $fs $ask)+ 批判($desc $fs $draft)──
    "council.view_user": """【场景描述】$desc
【few-shot 文档】$fs

$ask 严格 JSON。""",

    "council.critic_user": """【场景描述】$desc
【few-shot 文档】$fs

【代码装配的白皮书草案】$draft
挑硬伤,产出修订版完整白皮书 JSON(保留 active_lines/domain_profile/shared_world_spec 等全部字段)。""",
}


def render(name: str, **vars) -> str:
    """渲染一个 prompt。$var 占位;JSON `{}` 保持字面。缺失的 $var 原样保留(safe_substitute)。"""
    tmpl = PROMPTS.get(name)
    if tmpl is None:
        raise KeyError(f"未知 prompt: {name!r};可用:{sorted(PROMPTS)}")
    return Template(tmpl).safe_substitute(**vars)
