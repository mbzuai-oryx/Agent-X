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
| `OPENAI_API_KEY` | GPT-4o judge | only if `judge: gpt`. To judge with a non-OpenAI endpoint set `AGENTX_JUDGE_API_BASE` + `AGENTX_JUDGE_MODEL` too — the key var is read regardless of provider. |
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

## 6. Judge-free run on a limited number of records

The cheapest useful run: N tasks, no judge, no OpenAI key. Everything below is
already the default in [config_agentx.yaml](config_agentx.yaml).

```yaml
agentx_options:
  limit: 5            # first N tasks of dataset.json (0 = all 828)
  run_eval: false     # skip the judge entirely
  stop: ""            # SambaNova-hosted gemma/llama: no ChatML stop token
```

`limit` truncates `dataset.json` in place for the run and restores it from
`dataset.json.suite-bak` in a `finally` block, so an interrupted run leaves the
full dataset behind. Keys are taken in order (`"0"`..`"N-1"`) and align with
`data.json`, so the same subset can be judged later without re-running inference.

Steps (from the repo root, on a node that can reach the tool server):

```bash
CONDA=/import/snvm-sc-scratch2/rodrigom/miniforge3
source $CONDA/etc/profile.d/conda.sh

# 1. tool server — leave running in its own tmux window
conda activate agentlego
cd tests/agentx_snova/agentlego
export $(grep -E '^(SERPER|MATHPIX)' ../.env | xargs)
agentlego-server start --port 16181 --device cpu --no-setup --extra ./benchmark.py \
    Calculator OCR Plot Solver GoogleSearch MathOCR --host 0.0.0.0

# 2. confirm it is up — the run dies at model-build time if this 404s/refuses
curl -s localhost:16181/openapi.json | head -c 200

# 3. inference only
conda activate opencompass
cd tests/agentx_snova
python run_agentx.py --dry-run    # check the injected model dict
python run_agentx.py
```

Result: `logs/agentx/<ts>/{provider}/{model}/preds.json` plus a `[SUMMARY]` line
reporting how many of the N tasks produced a final answer, used tools, or hit
step errors. **No CSV and no S3 report** — `agentx_report.py` averages judge
metrics, so it only runs when `run_eval: true`.

## 7. Turning the judge on

Verified working end-to-end against the existing `venv_agentx_judge` — no code
changes needed. To score a full run:

```yaml
agentx_options:
  run_eval: true
  judge: gpt
  judge_python: "venv_agentx_judge/bin/python"   # openai==0.28.0 lives here
```

and in `.env`, uncomment the AWS block (otherwise the CSV is written locally and
the S3 upload logs `[WARN] Upload failed`, so nothing reaches Athena/Grafana):

```
AWS_REGION=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_S3_BUCKET_NAME=...
```

The judge env is separate on purpose: `evaluation/multiagent_evaluation.py` uses
`openai.ChatCompletion.create`, removed in openai 1.x, which the inference env
needs. `run_agentx.py` spawns it as a subprocess so the keys are inherited from
the already-loaded `.env` — the judge env itself has no `python-dotenv`.

It lives at `tests/agentx_snova/venv_agentx_judge/` (a conda **prefix** env, not
a `python -m venv`, ~243MB) and is git-ignored via this submodule's own
`.gitignore`. To rebuild it from scratch:

```bash
conda create -p tests/agentx_snova/venv_agentx_judge python=3.10 -y
tests/agentx_snova/venv_agentx_judge/bin/pip install 'openai==0.28.0' tqdm
```

Conda prefix envs are not relocatable: `bin/python` resolves its prefix from its
own location and survives a move, but the ~15 shebang'd scripts in `bin/` (`pip`,
`openai`, `wheel`, `*-config`, ...) hardcode the old absolute path. If you move
the directory, rewrite them and update the conda registry:

```bash
OLD=<old abs path>; NEW=<new abs path>
grep -rl "$OLD" "$NEW/bin" | xargs sed -i "s|$OLD|$NEW|g"
sed -i "s|^$OLD\$|$NEW|" ~/.conda/environments.txt
```

Judge model / endpoint are env-controlled: `AGENTX_JUDGE_MODEL` (default
`gpt-4o`) and `AGENTX_JUDGE_API_BASE` (unset ⇒ `api.openai.com`; set it without
the `/chat/completions` suffix to judge on any OpenAI-compatible provider).

To score an already-completed infer-only run without re-running inference:

```bash
set -a; . ./.env; set +a
cd evaluation && ../venv_agentx_judge/bin/python run_eval_gpt_as_judge.py \
    --pred_path  ../logs/agentx/<ts>/SambaNova/<model>/preds.json \
    --gt_data_path ../opencompass/data/agentx_dataset/data.json \
    --save_path  ../logs/agentx/<ts>/SambaNova/<model>/scores.json
cd .. && python agentx_report.py logs/agentx/<ts> fc-so-testing-suite/agentx_snova/<ts>
```

### Judge cost, runtime, and failure modes

* **12 sequential gpt-4o calls per task**, no batching, no concurrency: ~1.7k
  input + ~200 output tokens each, so roughly **$0.07/task** — a few cents for a
  5-task smoke test, order **$50–60 and many hours for all 828**.
* **No retry / rate-limit handling.** A single failed call is swallowed by a
  broad `except Exception`, leaving that task's 12 metrics `None`.
  `agentx_report.py` excludes `None` from each metric's denominator, so a
  partially rate-limited run silently averages over fewer tasks than
  `num_tasks` claims. Check `judge.log` for tracebacks before trusting a row.
* **Metric output is free-form**, not numeric: each metric comes back as a
  string containing a Python dict, sometimes inside a ```` ```python ```` fence,
  with `Score` as `'0.25'`, `0.0`, or an apostrophe-broken literal.
  `agentx_report._extract_score` handles all of these (literal_eval → bare float
  → regex fallback); verified all 12 parse on real gpt-4o output.
* **8 of the 12 metrics score the reasoning *trace*, not the answer.** A model
  that answers directly without tool calls scores ~0 on grounding, precision,
  tool/toolset accuracy, faithfulness, context and reward even when its final
  answer is right. Confirm the `[SUMMARY]` line shows real tool use before
  reading anything into low scores.

### Tool fidelity vs. hardware

`tool_meta` registers a `DummyTool` (returns the literal `'Dummy Result'`) for
every one of the 14 tools the server does *not* serve. The CPU command above
serves only the 6 non-vision tools; `ImageDescription`, `CountGivenObject`,
`RegionAttributeDescription`, `TextToBbox`, `DrawBox`, `AddText`, `TextToImage`
and `ImageStylization` need a GPU (Qwen-VL-Chat / mmdet) — see
[validate_on_gpu.sh](validate_on_gpu.sh) for the full-fidelity `--device cuda`
command. **Vision-centric metrics from a CPU-only server are not comparable to
published Agent-X numbers.** Setting `tool_server: ""` drops the server entirely
and makes every tool a stub — plumbing check only.

## Verified prediction format (was: "verify on first real run")

`run_agentx.py::consolidate_predictions()` converts OpenCompass's on-disk
predictions into the judge's `--pred_path` format. Confirmed against a real run:
`predictions/<abbr>/Agent-X.json` is a dict keyed by task id (`"0"`, `"1"`, ...)
with `{gold, prediction, origin_prompt, steps}`, and

* `steps` is **always `[]`** — `AgentInferencerOutputHandler.
  save_multiround_results()` seeds the key but never appends to it. The real
  ReAct trace (thought + `tool_calls` + tool results) is in `prediction`, which
  is a list of rounds of step dicts. `reasoning_steps` is taken from there.
* only the **last** assistant `content` is the final answer; earlier `content`
  fields are tool results (OCR dumps, boxes) and must not be folded into it.
