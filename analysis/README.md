# Agent-X Evaluation Notebooks

This directory contains two Jupyter notebooks used for evaluating and analyzing the performance of models on the **Agent-X benchmark**. Each notebook serves a distinct purpose in the evaluation pipeline.

---

## 1. `metrics.ipynb`: Evaluation Metrics Computation

This notebook computes **quantitative performance metrics** for vision-language models on the Agent-X dataset.

### Key Features

- Loads model predictions and ground truth reasoning traces
- Computes various evaluation metrics including:
  - **Step Accuracy**
  - **Tool Usage Accuracy**
  - **Final Answer Accuracy**
  - **Deep Reasoning Quality Scores**
- Supports evaluation under multiple modes:
  - Step-by-step
  - Deep Reasoning
  - Outcome (final answer correctness)

### Use Case

Use this notebook when you want to **quantify model performance** across multiple reasoning dimensions.

---

## 2. `error_analysis.ipynb`: Qualitative Error Categorization

This notebook supports **manual and automatic error analysis** for model-generated reasoning traces.

### Key Features

- Categorizes errors into:
  - Planning
  - Tool misuse
  - Format inconsistency
  - Logical incoherence
- Highlights where the model failed in step-by-step traces
- Aggregates error types to analyze dominant failure modes

### Use Case

Use this notebook when you want to **understand why** a model is failing and what kinds of reasoning or planning errors are most frequent.


---

## Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- `pandas`, `openpyxl`, `matplotlib`, and any other packages noted in the first cells of the notebooks

---

## Notes

- These notebooks assume you have the model outputs and ground truth data available in a structured format (e.g., `.csv` or `.xlsx`).
- For best results, use them in conjunction with the `generation/` folder that provides the query and reasoning generation scripts.
