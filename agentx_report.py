"""Aggregate Agent-X judge scores into a semicolon-delimited CSV and upload to S3.

Walks logs/agentx/<ts>/{provider}/{model}/scores.json (produced by the judge in
run_agentx.py), averages the 12 per-task metrics into a single row per model, and
writes one CSV matching the `agentx_results` Athena table in data_prep_athena.py.

The judge (evaluation/run_eval_gpt_as_judge.py) emits, per task key:
    {grounding_accuracy, precision_score, tool_accuray, faithfulness_accuray,
     goal_accuray, toolset_accuray, step_score, context_score, clarity_penalty,
     factual_precision, semantic_accuracy, reward_score}
We keep the upstream (occasionally misspelled) keys for parsing but write clean
column names to the CSV.
"""

import os
import ast
import csv
import re
import sys
import json
from pathlib import Path
from datetime import datetime

import boto3
from dotenv import load_dotenv

# raw judge key -> clean CSV/Athena column name (order defines CSV column order).
METRICS = [
    ("grounding_accuracy", "grounding_accuracy"),
    ("precision_score", "precision_score"),
    ("tool_accuray", "tool_accuracy"),
    ("faithfulness_accuray", "faithfulness_accuracy"),
    ("goal_accuray", "goal_accuracy"),
    ("toolset_accuray", "toolset_accuracy"),
    ("step_score", "step_score"),
    ("context_score", "context_score"),
    ("clarity_penalty", "clarity_penalty"),
    ("factual_precision", "factual_precision"),
    ("semantic_accuracy", "semantic_accuracy"),
    ("reward_score", "reward_score"),
]
CLEAN_COLS = [clean for _, clean in METRICS]


def _extract_score(val):
    """Pull a numeric score (0..1) out of a judge metric value.

    The GPT judge returns each metric not as a number but as a dict — or, more
    often, a STRING repr of one — like "{'Score': '0.7', 'Justification': ...}",
    sometimes wrapped in ```python / ```json fences. Handle all those shapes;
    return None if no score can be recovered.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        return _extract_score(val.get("Score"))
    if isinstance(val, str):
        t = val.strip()
        # strip a leading ```lang fence and trailing ```
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
        # try to parse a python-literal dict (single quotes -> not JSON)
        try:
            obj = ast.literal_eval(t)
            if isinstance(obj, dict):
                return _extract_score(obj.get("Score"))
            if isinstance(obj, (int, float)):
                return float(obj)
        except (ValueError, SyntaxError):
            pass
        # bare number as a string
        try:
            return float(t)
        except ValueError:
            pass
        # last resort: regex the Score value out of the text
        m = re.search(r"['\"]?Score['\"]?\s*:\s*['\"]?(-?[0-9]*\.?[0-9]+)", t)
        if m:
            return float(m.group(1))
    return None


def _load_env():
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def average_scores(scores_path):
    """Mean of each metric across tasks, ignoring None/non-numeric. Returns
    (means_by_clean_col, num_tasks)."""
    with open(scores_path) as f:
        data = json.load(f)  # {task_key: {raw_metric: value|None}}

    sums = {raw: 0.0 for raw, _ in METRICS}
    counts = {raw: 0 for raw, _ in METRICS}
    for _task, metrics in data.items():
        if not isinstance(metrics, dict):
            continue
        for raw, _clean in METRICS:
            score = _extract_score(metrics.get(raw))
            if score is not None:
                sums[raw] += score
                counts[raw] += 1

    means = {}
    for raw, clean in METRICS:
        means[clean] = round(sums[raw] / counts[raw], 4) if counts[raw] else None
    return means, len(data)


def _upload_to_s3(local_path, s3_prefix):
    try:
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


def generate_report(logs_dir, s3_prefix=None):
    logs_path = Path(logs_dir)
    rows = []

    for provider_dir in sorted(p for p in logs_path.iterdir() if p.is_dir()):
        provider = provider_dir.name
        for model_dir in sorted(p for p in provider_dir.iterdir() if p.is_dir()):
            model = model_dir.name
            scores_path = model_dir / "scores.json"
            if not scores_path.exists():
                print(f"  {provider}/{model}: no scores.json")
                continue
            try:
                means, num_tasks = average_scores(scores_path)
            except Exception as e:
                print(f"  {provider}/{model}: failed to parse scores.json: {e}")
                continue
            row = {
                "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                "provider": provider,
                "model": model,
                "num_tasks": num_tasks,
                **means,
            }
            rows.append(row)
            print(f"  {provider}/{model}: {num_tasks} tasks | "
                  f"goal={means.get('goal_accuracy')} sem={means.get('semantic_accuracy')}")

    if not rows:
        print("[WARN] No results found in", logs_dir)
        return None

    fieldnames = ["date", "provider", "model", "num_tasks"] + CLEAN_COLS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    csv_path = logs_path / f"results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[REPORT] CSV written to: {csv_path}")
    if s3_prefix:
        _upload_to_s3(csv_path, s3_prefix)
    return str(csv_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: agentx_report.py <logs_dir> [s3_prefix]")
        sys.exit(1)
    _load_env()
    generate_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
