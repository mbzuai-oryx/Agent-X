# Agent-X — Setup & Run (FC-SO Testing Suite integration)

Agent-X (https://github.com/mbzuai-oryx/Agent-X) is a **vision-centric agentic
tool-use** benchmark: 828 multimodal tasks across 6 domains (web, surveillance,
driving, sports, math, data). It has two stages — OpenCompass inference against
an AgentLego tool server, then an LLM-as-judge that emits 12 metrics per task.

This directory adds the suite wrappers: [config_agentx.yaml](config_agentx.yaml),
[run_agentx.py](run_agentx.py), [agentx_report.py](agentx_report.py). They inject
provider endpoints into `opencompass/configs/eval_gta_bench.py`, run inference,
run the judge, and aggregate results to a CSV → S3 (`agentx_results` Athena table).

> **Models must be multimodal.** Text-only models cannot solve the tasks — mark
> them `not_available` in `model_mappings`.

## 1. One-time environment (GCP VM `benchmarking-client-dev`)

Agent-X uses conda (not the suite's `uv`/venv), two envs:

```bash
# Tool server env
conda create -n agentlego python=3.11.9 -y && conda activate agentlego
cd tests/agentx_snova/agentlego
pip install -r requirements_all.txt && pip install agentlego && pip install -e .
mim install mmengine && mim install mmcv==2.1.0
# Edit transformers/modeling_utils.py line ~1279: _supports_sdpa = False -> True

# Inference env
conda create -n opencompass python=3.10 -y && conda activate opencompass
cd tests/agentx_snova/agentlego && pip install -e .
cd ../opencompass && pip install -e .
```

We do **NOT** need LMDeploy — models under test are hosted provider APIs
(OpenAI-compatible), not locally-served HF weights.

## 2. Dataset

Download from https://huggingface.co/datasets/Tajamul21/Agent-X into:

```
tests/agentx_snova/opencompass/data/agentx_dataset/
├── dataset.json      # ground truth (judge --gt_data_path)
├── toolmeta.json
└── image/            # all images/videos here
```

## 3. API keys — `tests/agentx_snova/.env`

See [.env.example](.env.example). For a **SambaNova-only smoke test** you need:

| Key | Needed for | Notes |
|-----|-----------|-------|
| `SAMBANOVA_API_KEY` | model under test | already in the suite |
| `SERPER_API_KEY` | GoogleSearch tool | https://serper.dev (free tier) — skip if avoiding web-search tasks |
| `MATHPIX_APP_ID` / `MATHPIX_APP_KEY` | MathOCR tool | https://mathpix.com — skip if avoiding math tasks |
| `OPENAI_API_KEY` | GPT-4o judge | only if `judge: gpt`. Reuse SambaNova as judge to skip. |
| `AWS_*` | S3 upload | standard suite vars |

Cheapest first run: `run_eval: false` (infer only, no judge key) on a small task
subset that avoids Serper/Mathpix.

## 4. Start the AgentLego tool server (env: agentlego)

```bash
conda activate agentlego
export SERPER_API_KEY=... MATHPIX_APP_ID=... MATHPIX_APP_KEY=...
cd tests/agentx_snova/agentlego
agentlego-server start --port 16181 --device cpu --no-setup --extra ./benchmark.py OCR --host 0.0.0.0
```

Leave it running (use tmux). Matches `agentx_options.tool_server` in the config.

## 5. Run (env: opencompass)

```bash
conda activate opencompass
cd tests/agentx_snova
python run_agentx.py --dry-run   # prints injected config + commands, runs nothing
python run_agentx.py             # infer -> judge -> CSV -> S3
```

Outputs: `logs/agentx/<ts>/{provider}/{model}/{preds.json,scores.json}` and
`results_<ts>.csv`, uploaded to `s3://.../fc-so-testing-suite/agentx_snova/<ts>/`.

## Output & metrics

Each of the 12 metrics is a separate GPT/Qwen-judge call returning
`{'Score': <0–1>, 'Justification': ...}` per task. `agentx_report.py` parses the
`Score`, **averages each metric across all tasks**, and writes one CSV row per
model: `date, provider, model, num_tasks` + the 12 averaged scores (each 0–1).
The CSV feeds the `agentx_results` Athena table; the unified Grafana view surfaces
**`goal_accuracy`** as the cross-benchmark headline (the other 11 stay queryable
in `agentx_results`).

| Metric | Meaning (all 0–1) |
|--------|-------------------|
| **goal_accuracy** ⭐ | Final-answer correctness — cosine similarity to GT (or 0/1 for exact-answer types). **Headline metric.** |
| grounding_accuracy | Per step: reasoning grounded in the actual visual/tool evidence (avg per-step) |
| precision_score | Binary 0/1 — precision of the reasoning vs GT |
| tool_accuracy | Correct tools called vs GT `tool_metadata` |
| toolset_accuracy | F1 over the *set* of tools used vs GT |
| faithfulness_accuracy | Reasoning trace logically faithful to the GT plan |
| step_score | Quality of each reasoning step (avg per-step) |
| context_score | Each step uses available context appropriately (avg per-step) |
| factual_precision | Factual correctness of claims in the reasoning vs GT |
| semantic_accuracy | Semantic match of reasoning + final answer to GT |
| reward_score | Self-correction ability — recognizing and fixing its own mistakes |
| clarity_penalty | Penalty for unclear/verbose reasoning (**higher = worse**, unlike the rest) |

Metric definitions live in `evaluation/multiagent_evaluation.py` (`get_*` fns).

## ⚠️ Verify on first real run

`run_agentx.py::consolidate_predictions()` converts OpenCompass's on-disk
predictions into the judge's `--pred_path` format. The field layout is now
verified against the vendored inferencer: with `infer_mode='every'`
(`configs/datasets/gta_bench.py`), `AgentInferencer.save_multiround_results`
writes each task as `{"gold", "prediction": [[step, …]], "origin_prompt",
"steps": []}` — the reasoning trace is in `prediction` (a list of turns, each a
list of `{role, content|tool_calls}` step dicts) and `steps` is always `[]`.
Accordingly we take `reasoning_steps` from `prediction` and `final_answer` from
the last assistant-content step.

Still worth a spot-check on the first run: inspect a file under
`opencompass/outputs/default/<ts>/predictions/<abbr>/Agent-X.json` and confirm the
task keys match `data.json` (both keyed "0","1",…) and that a finish-action
answer is present. The function raises loudly on an empty/unparseable result
rather than emitting silently-wrong scores.
