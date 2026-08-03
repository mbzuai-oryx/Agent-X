"""Run the Agent-X benchmark across providers for the FC-SO testing suite.

Agent-X is a vision-centric agentic tool-use benchmark. Unlike the other suites
it has TWO stages driven by different tools:

  1. Inference  — OpenCompass runs a ReAct agent (`LagentAgent` + `OpenAI` llm)
                  against the AgentLego tool server. The model under test is a
                  dict in `opencompass/configs/eval_gta_bench.py::models`. We
                  inject one dict per enabled provider/model (webbench-style
                  backup/restore of the config), then run a single
                  `python run.py ... --mode infer`.
  2. Judging    — a standalone GPT-4o / Qwen judge scores each prediction file
                  and emits 12 metrics per task (`evaluation/run_eval_*_judge.py`).

Then agentx_report.py averages the metrics per model into a CSV and uploads it.

PREREQUISITES (see AGENTX_SETUP.md):
  * conda env `opencompass` active (this script is launched from within it)
  * AgentLego tool server running at agentx_options.tool_server
  * dataset downloaded to opencompass/data/agentx_dataset/
  * .env with provider keys (+ OPENAI_API_KEY if judge: gpt)
"""

import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

from agentx_report import generate_report

CONFIG_FILENAME = "config_agentx.yaml"
BENCH_GROUP_NAME = "agentx"

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
config_path = current_dir / CONFIG_FILENAME
env_file = current_dir / ".env"

# Agent-X vendored sub-projects (relative to the submodule root).
OPENCOMPASS_DIR = current_dir / "opencompass"
EVAL_CONFIG = OPENCOMPASS_DIR / "configs" / "eval_gta_bench.py"
JUDGE_SCRIPTS = {
    "gpt": current_dir / "evaluation" / "run_eval_gpt_as_judge.py",
    "qwen": current_dir / "evaluation" / "run_eval_qwen_as_judge.py",
}


def load_env():
    print(f"[SETUP] Loading env from {env_file}")
    if not env_file.exists():
        print(f"[ERROR] {env_file} not found.")
        sys.exit(1)
    load_dotenv(env_file, override=True)
    print("[OK] Environment variables loaded.\n")


def run_command(cmd, log_file, cwd=None, dry_run=False):
    """Run a command, tee combined output to log_file, return True on success."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[RUNNING] (cwd={cwd or os.getcwd()}) {cmd_str}")
    if dry_run:
        print("[DRY-RUN] Skipping execution.")
        return True
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                [str(c) for c in cmd], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, cwd=cwd, check=False,
            )
            lf.write(result.stdout)
            print(result.stdout.strip())
        ok = result.returncode == 0
        print(f"[{'DONE' if ok else 'FAIL'}] exit={result.returncode}: {cmd_str}")
        return ok
    except Exception as e:
        print(f"[EXCEPTION] {cmd_str}: {e}")
        return False


def upload_file(local_path, s3_prefix):
    try:
        import boto3

        bucket = os.environ.get("AWS_S3_BUCKET_NAME", "")
        s3_key = f"{s3_prefix}/{local_path.name}"
        s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", ""),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"[UPLOAD] {local_path} -> s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"[WARN] Upload failed for {local_path}: {e}")


def abbr_for(provider, alias):
    """OpenCompass output dir name for a model. Kept filesystem-safe and
    reversible so the report can recover provider/alias."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{provider}__{alias}")
    return safe


def build_models(cfg):
    """Build the list of OpenCompass model dicts (as source text) for every
    enabled provider/model, plus a manifest of (provider, alias, abbr)."""
    opts = cfg["agentx_options"]
    base_urls = cfg.get("base_urls", {})
    provider_api_keys = cfg.get("provider_api_keys", {})
    model_mappings = cfg.get("model_mappings", {})

    model_dicts, manifest = [], []
    # `stop` is model-family specific ('<|im_end|>' is ChatML/Qwen). Sending it to
    # a provider that doesn't use it leaves the raw fence/EOS text in the answer,
    # so it is opt-in via config instead of hardcoded.
    stop = opts.get("stop") or None
    # tool_server is optional: when blank, LagentAgent registers only the
    # DummyTool stubs from tool_meta (every call returns 'Dummy Result'). That
    # validates plumbing without the AgentLego server but produces meaningless
    # scores — never use it for a reportable run.
    tool_server = opts.get("tool_server") or None

    for provider, models in model_mappings.items():
        base_url = base_urls.get(provider)
        key_env = provider_api_keys.get(provider, "")
        api_key = os.environ.get(key_env, "") if key_env else ""
        if not base_url:
            print(f"[WARN] No base_url for {provider}; skipping.")
            continue
        if not api_key:
            print(f"[WARN] Env var {key_env} empty for {provider}; skipping.")
            continue
        for alias, model_id in models.items():
            if model_id == "not_available":
                print(f"[SKIP] {provider}/{alias}")
                continue
            abbr = abbr_for(provider, alias)
            model_dicts.append(
                "    dict(\n"
                f"        abbr={abbr!r},\n"
                "        type=LagentAgent,\n"
                "        agent_type=ReAct,\n"
                f"        max_turn={int(opts.get('max_turn', 10))},\n"
                "        llm=dict(\n"
                "            type=OpenAI,\n"
                f"            path={model_id!r},\n"
                f"            key={api_key!r},\n"
                f"            openai_api_base={base_url!r},\n"
                f"            query_per_second={int(opts.get('query_per_second', 1))},\n"
                f"            max_seq_len={int(opts.get('max_seq_len', 4096))},\n"
                + (f"            stop={stop!r},\n" if stop else "")
                + "        ),\n"
                + (f"        tool_server={tool_server!r},\n" if tool_server else "")
                + f"        tool_meta={opts['tool_meta']!r},\n"
                f"        batch_size={int(opts.get('batch_size', 8))},\n"
                "    ),"
            )
            manifest.append({"provider": provider, "alias": alias, "abbr": abbr,
                             "model_id": model_id})
    return model_dicts, manifest


def render_eval_config(model_dicts):
    """Full source of a generated eval_gta_bench.py with our models injected."""
    body = "\n".join(model_dicts)
    return (
        "# AUTO-GENERATED by run_agentx.py — do not edit. Original backed up as\n"
        "# eval_gta_bench.py.suite-bak and restored after the run.\n"
        "from lagent.agents import ReAct\n"
        "from mmengine.config import read_base\n"
        "from opencompass.models import OpenAI, Qwen, Gemini\n"
        "from opencompass.models.lagent import LagentAgent\n"
        "from opencompass.partitioners import SizePartitioner\n"
        "from opencompass.runners import LocalRunner\n"
        "from opencompass.tasks import OpenICLInferTask\n\n"
        "with read_base():\n"
        "    from .datasets.gta_bench import gta_bench_datasets as datasets\n\n"
        f"models = [\n{body}\n]\n\n"
        "infer = dict(\n"
        "    partitioner=dict(type=SizePartitioner, max_task_size=50, gen_task_coef=1),\n"
        "    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),\n"
        ")\n"
    )


def truncate_dataset(dataset_path, limit):
    """Back up dataset.json and rewrite it with only the first `limit` tasks, so a
    smoke test runs a handful of tasks instead of all 828. Returns the backup path
    (to restore later), or None if no truncation happened.

    dataset.json is a dict keyed by task id (per the judge: gt_data[key][0][...]).
    Falls back to list slicing if the top level is a list.
    """
    if not limit or limit <= 0:
        return None
    if not dataset_path.exists():
        print(f"[WARN] dataset not found at {dataset_path}; cannot apply limit.")
        return None

    with open(dataset_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        subset = dict(list(data.items())[:limit])
    elif isinstance(data, list):
        subset = data[:limit]
    else:
        print(f"[WARN] Unexpected dataset.json type {type(data)}; skipping limit.")
        return None

    backup = dataset_path.with_suffix(".json.suite-bak")
    shutil.copy2(dataset_path, backup)
    with open(dataset_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"[LIMIT] Truncated dataset to {len(subset)} task(s); "
          f"original backed up as {backup.name}")
    return backup


def latest_output_dir():
    """Newest OpenCompass run dir under opencompass/outputs/default/."""
    outputs = OPENCOMPASS_DIR / "outputs" / "default"
    if not outputs.exists():
        return None
    runs = sorted((p for p in outputs.iterdir() if p.is_dir()), key=lambda p: p.name)
    return runs[-1] if runs else None


def _flatten_steps(prediction):
    """Flatten an OpenCompass `prediction` into its ordered list of ReAct steps.

    `prediction` is a list of rounds, each round a list of step dicts:
        [[{'role': 'assistant', 'thought': ..., 'tool_calls': [...]},
          {'role': 'tool', 'name': 'OCR', 'content': ...},
          {'role': 'assistant', 'content': 'final answer'}]]
    """
    steps = []

    def _walk(node):
        if isinstance(node, dict):
            steps.append(node)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(prediction)
    return steps


def _clean_answer(text):
    """Strip stray markdown fences the ReAct protocol leaves on the answer."""
    return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip()).strip()


def _extract_final_answer(prediction):
    """Pull the final assistant answer text out of an OpenCompass `prediction`.

    Only the LAST assistant `content` is the answer. Earlier `content` fields in
    the trace belong to tool results (OCR dumps, detection boxes, ...) and must
    not be concatenated into it — doing so pollutes the judge's goal_accuracy and
    semantic_accuracy inputs with tool output.
    """
    if isinstance(prediction, str):
        return _clean_answer(prediction)

    steps = _flatten_steps(prediction)
    for step in reversed(steps):
        if step.get("role") == "assistant" and isinstance(step.get("content"), str):
            return _clean_answer(step["content"])
    # No assistant answer (e.g. the run ended on an error step): fall back to the
    # last textual content of any role rather than dropping the task silently.
    for step in reversed(steps):
        if isinstance(step.get("content"), str):
            return _clean_answer(step["content"])
    return ""


def consolidate_predictions(work_dir, abbr, dest):
    """Convert OpenCompass predictions for one model into the judge's format.

    The judge (evaluation/run_eval_gpt_as_judge.py) expects `--pred_path` to be a
    JSON list where each element is {task_key: {"reasoning_steps": ...,
    "final_answer": ...}}, keyed by the same task id as data.json ("0", "1", ...).

    OpenCompass writes predictions/<abbr>/Agent-X.json (final) or tmp_Agent-X.json
    (partial run) as a dict keyed by task id, each entry:
        {"gold": ..., "prediction": [[...]], "origin_prompt": ..., "steps": ...}

    VERIFIED against a real run (opencompass/outputs/default/*/predictions/):
    `steps` is ALWAYS an empty list — AgentInferencerOutputHandler.
    save_multiround_results() seeds it but never appends. The actual ReAct trace
    (thoughts + tool_calls + tool results) is what lives in `prediction`, so
    reasoning_steps is taken from `prediction` and `steps` is only a fallback.
    """
    pred_dir = work_dir / "predictions" / abbr
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"No predictions dir for {abbr} at {pred_dir}. Did inference run/finish?"
        )

    # Prefer the completed Agent-X.json over any tmp_ partial file.
    files = sorted(pred_dir.glob("*.json"))
    final_files = [f for f in files if not f.name.startswith("tmp_")]
    use_files = final_files or files

    merged = {}
    for pf in use_files:
        with open(pf) as f:
            data = json.load(f)
        for task_key, item in data.items():
            if not isinstance(item, dict):
                continue
            prediction = item.get("prediction")
            steps = item.get("steps") or _flatten_steps(prediction)
            merged[str(task_key)] = {
                "reasoning_steps": steps,
                "final_answer": _extract_final_answer(prediction),
            }

    if not merged:
        raise ValueError(f"No predictions parsed for {abbr} in {pred_dir}.")

    pred_list = [{k: v} for k, v in merged.items()]
    with open(dest, "w") as f:
        json.dump(pred_list, f, indent=2)
    print(f"[PRED] {abbr}: {len(pred_list)} tasks -> {dest}")
    return dest


def summarize_predictions(preds_path):
    """Print trace stats for an infer-only run (run_eval: false).

    Without a judge there is no score, so this is the only signal that inference
    actually exercised the agent loop: a run where every task has 0 tool calls
    means the tool server or the ReAct protocol is not being used, no matter how
    cleanly OpenCompass exited.
    """
    with open(preds_path) as f:
        pred_list = json.load(f)

    n = len(pred_list)
    with_answer = with_tools = with_error = 0
    total_calls = 0
    for entry in pred_list:
        value = next(iter(entry.values()))
        steps = value.get("reasoning_steps") or []
        calls = sum(len(s.get("tool_calls") or []) for s in steps
                    if isinstance(s, dict))
        total_calls += calls
        with_tools += calls > 0
        with_error += any(isinstance(s, dict) and s.get("error") for s in steps)
        with_answer += bool(value.get("final_answer"))

    print(f"[SUMMARY] {n} task(s) | final_answer: {with_answer}/{n} | "
          f"used tools: {with_tools}/{n} ({total_calls} calls) | "
          f"step errors: {with_error}/{n}")
    if n and not with_tools:
        print("[WARN] No tool calls in any trace — check that the AgentLego tool "
              "server is reachable and that agentx_options.tool_server is set.")


def main():
    dry_run = "--dry-run" in sys.argv

    load_env()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    opts = cfg["agentx_options"]
    judge = opts.get("judge", "gpt")
    run_eval = opts.get("run_eval", True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = current_dir / "logs" / BENCH_GROUP_NAME / run_timestamp
    output_base.mkdir(parents=True, exist_ok=True)
    s3_prefix = f"fc-so-testing-suite/agentx_snova/{run_timestamp}"

    print(f"[INFO] Local output: {output_base}")
    print(f"[INFO] S3 prefix:   s3://{os.getenv('AWS_S3_BUCKET_NAME')}/{s3_prefix}\n")

    # --- Build + inject the model list into eval_gta_bench.py ---------------
    model_dicts, manifest = build_models(cfg)
    if not manifest:
        print("[ERROR] No runnable models. Check base_urls / API keys / mappings.")
        sys.exit(1)

    print(f"[INFO] {len(manifest)} model(s) to run:")
    for m in manifest:
        print(f"   - {m['provider']}/{m['alias']} (abbr={m['abbr']})")

    limit = opts.get("limit", 0)
    # Truncate the INFERENCE dataset (dataset.json) to limit how many tasks run,
    # NOT the judge ground_truth (data.json) — different files, different schemas.
    dataset_path = OPENCOMPASS_DIR / opts.get("infer_dataset", opts["ground_truth"])

    backup = EVAL_CONFIG.with_suffix(".py.suite-bak")
    dataset_backup = None
    if dry_run:
        print(f"\n[DRY-RUN] Would back up {EVAL_CONFIG} -> {backup}")
        if limit and limit > 0:
            print(f"[DRY-RUN] Would truncate {dataset_path} to first {limit} task(s)")
        print("[DRY-RUN] Would write generated config:\n")
        print(render_eval_config(model_dicts))
    else:
        shutil.copy2(EVAL_CONFIG, backup)
        EVAL_CONFIG.write_text(render_eval_config(model_dicts))
        print(f"[OK] Injected {len(manifest)} model(s) into {EVAL_CONFIG}")
        dataset_backup = truncate_dataset(dataset_path, limit)

    try:
        # --- Inference (OpenCompass) --------------------------------------
        infer_cmd = [
            "python", "run.py", "configs/eval_gta_bench.py",
            "--max-num-workers", str(opts.get("max_num_workers", 8)),
            "--debug", "--mode", "infer",
        ]
        run_command(infer_cmd, output_base / "infer.log",
                    cwd=OPENCOMPASS_DIR, dry_run=dry_run)

        work_dir = latest_output_dir() if not dry_run else None
        if not dry_run:
            if work_dir is None:
                print("[ERROR] No OpenCompass output dir found after inference.")
                sys.exit(1)
            print(f"[INFO] OpenCompass work dir: {work_dir}")

        # --- Per-model: judge + upload ------------------------------------
        for m in manifest:
            model_dir = output_base / m["provider"] / m["alias"]
            model_dir.mkdir(parents=True, exist_ok=True)
            preds_path = model_dir / "preds.json"
            scores_path = model_dir / "scores.json"

            if dry_run:
                print(f"[DRY-RUN] {m['provider']}/{m['alias']}: consolidate preds, "
                      f"then judge={judge} -> {scores_path.name}")
                continue

            try:
                consolidate_predictions(work_dir, m["abbr"], preds_path)
            except Exception as e:
                print(f"[WARN] {m['provider']}/{m['alias']} prediction consolidation "
                      f"failed: {e}")
                continue

            if run_eval:
                judge_script = JUDGE_SCRIPTS[judge]
                # The GPT judge needs openai==0.28.0, so it runs in its own venv
                # (config judge_python). Falls back to this interpreter if unset.
                jp = opts.get("judge_python") or sys.executable
                # abspath (not resolve): normalize '..' WITHOUT following the
                # venv's python symlink, else it'd run the system interpreter
                # and miss the judge venv's openai==0.28.0.
                jp_path = jp if os.path.isabs(jp) else os.path.abspath(
                    os.path.join(current_dir, jp))
                judge_cmd = [
                    jp_path, str(judge_script),
                    "--save_path", str(scores_path),
                    "--gt_data_path", str(OPENCOMPASS_DIR / opts["ground_truth"]),
                    "--pred_path", str(preds_path),
                ]
                run_command(judge_cmd, model_dir / "judge.log", dry_run=False)
            else:
                print(f"[SKIP-EVAL] {m['provider']}/{m['alias']} (run_eval=false)")
                summarize_predictions(preds_path)

            # Upload raw artifacts.
            model_s3 = f"{s3_prefix}/{m['provider']}/{m['alias']}"
            for artifact in [preds_path, scores_path, model_dir / "judge.log"]:
                if artifact.exists():
                    upload_file(artifact, model_s3)
    finally:
        # Always restore the original config + dataset.
        if not dry_run and backup.exists():
            shutil.move(str(backup), str(EVAL_CONFIG))
            print(f"[OK] Restored original {EVAL_CONFIG}")
        if dataset_backup and dataset_backup.exists():
            shutil.move(str(dataset_backup), str(dataset_path))
            print(f"[OK] Restored original {dataset_path}")

    if not dry_run and run_eval:
        generate_report(str(output_base), s3_prefix)

    print(f"\n[COMPLETE] Agent-X run finished.\nLocal: {output_base}"
          f"\nS3:    s3://{os.getenv('AWS_S3_BUCKET_NAME')}/{s3_prefix}")


if __name__ == "__main__":
    main()
