# Agent-X Evaluation Analysis

This folder contains post-processing notebooks for analyzing and evaluating model performance on the **Agent-X benchmark**. These tools help identify common failure modes and summarize performance across different reasoning modes, tools, and query types.

---

## 📊 1. `metrics.ipynb`

**Purpose:**  
Compute high-level statistics and breakdowns of the benchmark dataset and model outputs.

**What it includes:**
- Total number of queries and reasoning steps
- Breakdown by query type (factual, interpretive, generative)
- Tool usage frequency
- Distribution of steps per query
- Answer length and justification patterns
- (Optional) model-specific accuracy summaries

**Use this notebook to:**
- Understand dataset composition
- Compare different models' output trends
- Generate summary tables or plots for your paper

---

## 🧠 2. `error_analysis.ipynb`

**Purpose:**  
Analyze errors made by models in reasoning traces and final answers.

**What it includes:**
- Categorization of errors (e.g., planning, hallucination, tool misuse, factual mistakes)
- Counts and percentages of each error type per model
- Visual breakdowns of step-wise and outcome-level failures
- Examples of erroneous outputs and their types
- (Optional) correlation between number of steps and likelihood of failure

**Use this notebook to:**
- Diagnose where and why models fail
- Justify qualitative claims about failure types
- Prepare visualizations (e.g., pie charts or bar plots) for presentations or papers

---

## 📁 Expected Inputs

Both notebooks assume the availability of processed output files such as:
- `generated_agent_reasoning_*.xlsx` (model outputs)
- Categorized error CSVs (for `error_analysis`)
- Ground truth reference annotations

Make sure these are placed in a consistent directory (e.g., `./outputs/` or `./evaluation/`) and paths are updated in the notebook if needed.

---
