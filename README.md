# Animus

`codex/unverified-latest` 收录正在验证的生成与交付量控制改动。这个分支尚未完成端到端验收；已验收的发布代码在 `main`。

Animus 从结构化 seed 生成长期记忆 benchmark：构造世界、跨期语料和题目，逐题审查后导出标准包。命令行流程支持断点续跑；`--release` 会让四个原生选手试答，并剔除全员答对的题。

## 试答结果

两张图汇总生成期筛选样本中的 240 道题，覆盖法律、金融、鉴证和保险四个领域。图中列出四选手成绩、七项能力表现和题量分布。

![240 道题的四选手总成绩、七项能力雷达图和四领域成绩](assets/readme/01_performance_dashboard.png)

![四领域与七项能力的题量分布、四选手逐能力正确数和准确率](assets/readme/02_coverage_and_capabilities.png)

## 快速开始

需要 Python 3.10 及以上版本。四选手试答还需要 Node.js 22.19.0 及以上版本和 npm。以下命令在仓库根目录运行。

Windows PowerShell：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -X utf8 -m pip install -r requirements-minimal.txt
Copy-Item .env.example .env
Copy-Item configs/env/secrets.env.example configs/env/secrets.env
```

Linux/macOS：

```bash
python3 -m venv venv
./venv/bin/python -X utf8 -m pip install -r requirements-minimal.txt
cp .env.example .env
cp configs/env/secrets.env.example configs/env/secrets.env
```

安装原生选手 CLI：

```bash
npm install --global @openai/codex@0.153.2 @deepseek-ai/dsh@0.1.2-rc.1
```

在 `.env` 配置生成与裁判接口，在 `configs/env/secrets.env` 配置四选手接口。模型和协议见 [四选手配置](examples/release_four.json)。DSH 的固定 profile 见 [配置说明](configs/dsh/README.md)。

## 运行

通过 `--seed-pack` 指定自己的 seed JSON。输入格式见 [schema](schemas/seed_pack_v2.schema.json) 和 [转换指南](skills/realfiles-to-seedjson/SEEDJSON_GUIDE.md)。

```powershell
.\venv\Scripts\python.exe -X utf8 tools/validate_seed_packs.py path/to/seed.json --require-generation-ready
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --seed-pack path/to/seed.json --min-questions 200 --target-mchars 1 --haystack-ratio 9 --semantic-workers 4 --release
```

`--target-mchars 1` 指定 100 万字符的正文目标；`--haystack-ratio 9` 指定草堆与其余正文至少 9:1。Linux/macOS 将上面命令的解释器换成 `./venv/bin/python`。

中断后继续同一个 run：

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id>
```

已有完整生成结果时，可从质量汇总开始补做四选手试答和筛选：

```powershell
.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --run <run_id> --release --from quality
```

省略 `--release` 可单独运行生成部分。完整参数见 `.\venv\Scripts\python.exe -X utf8 -m pipeline.factory --help`。发布配置默认关闭 `L3_process_trace` 过程题；该题型的过程语义评分尚未接入。

按最终交付题量设置生成目标时，使用 `--delivery-target`。配置格式见 [交付量说明](DELIVERY_TARGET.md)。

## 产物与验证

运行结果保存在 `output/runs/<run_id>/`。`07_release.json` 记录质量审查结果，`08_calibration.json` 记录试答状态，`09_selection.json` 记录筛选结果；标准包位于 `delivery/<attempt>/benchmark/`。
