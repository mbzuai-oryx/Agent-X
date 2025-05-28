# Agent-X Evaluation Notebooks

This directory contains two Jupyter notebooks used for evaluating and analyzing the performance of models on the **Agent-X benchmark**. Each notebook serves a distinct purpose in the evaluation pipeline.

---

## 1. `metrics.ipynb`: Evaluation Metrics Computation

This notebook performs **quantitative evaluation** of model performance on the Agent-X benchmark, focusing on reasoning accuracy, tool usage correctness, and outcome quality.

It is structured to compute and visualize:

- Goal Accuracy
- Tool Metrics for Generative Queries
- Tool Call Success/Failure
- Reasoning Step Trends
- Difficulty-based Breakdown

---

## Goal Accuracy Computation

The notebook begins by computing **goal accuracy (G<sub>acc</sub>)** for each example.

### Step 1: Filter Generative Queries

We exclude generation-based examples (`GENERATIVE_IDS`) when computing the global goal accuracy because they follow a different evaluation scheme.

```python
# Clear goal_accuracy for generative rows
df.at[idx, "goal_accuray"] = ""
```

### Step 2: Global Average Accuracy

After filtering, the notebook computes the average `goal_accuray` across the rest of the dataset for a reliable benchmark.

### Step 3: Goal Score ★ for Generative Subset

Since generative queries don't have ground-truth answers, we approximate their goal success by averaging the following tool-based scores:

- `precision_score`
- `tool_accuray`
- `toolset_accuray`

This forms the **G<sub>a</sub><sup>*</sup> (Goal Accuracy Star)** metric.

```python
subset_means = {
    "precision_score": ...,
    "tool_accuray": ...,
    "toolset_accuray": ...
}
```

---

## Tool Call Statistics

We analyze how often tools were used successfully or failed across different models. This helps uncover issues like:

- Missing tool outputs
- Missing tool names
- Invalid tool calls

### Allowed Tool List

```python
allowed_tools = {
    "Calculator", "OCR", "ObjectCounter", "SceneDescriber",
    "WebSearch", "RegionDescriber", "LocateObjectByText",
    "CodePlotter", "MathOCR", "Solver", "DrawBoundingBox",
    "OverlayText", "ImageGenerator", "ImageStylization"
}
```

### Bar Chart: Tool Call Success vs Failures

![Tool Call Bar Chart](tool_call.png)

---

## Tool Usage Summary

For each JSON file of reasoning traces, the notebook extracts:

- Total reasoning steps
- Unique tools used
- Tool usage distribution

Saved as a `*.csv` file to compare models and enable trend plots.

```python
{
    "id": 43,
    "total_steps": 5,
    "unique_tools_used": 3,
    ...
}
```

---

## Trend: Goal Accuracy vs Tool Count / Step Depth

This part plots how reasoning **depth** and **tool diversity** affect performance.

```python
compare_models_goal_accuracy_trends([...])
```

### Goal Accuracy vs. Reasoning Steps

![Reasoning Chain Depth](goal_acc_vs_reasoning_steps.png)

---

## Difficulty-wise Goal Accuracy (GPT-4o Categorized)

We use a GPT-4o-generated categorization of query difficulty (`easy`, `medium`, `hard`) to plot how well models perform on hard vs. easy tasks.

```python
grouped = df.groupby("difficulty")["goal_accuray"].mean()
```

### Accuracy by Difficulty

![Goal Accuracy by Difficulty](difficulty.png)

---

## Summary

This notebook provides:

- A principled way to **isolate evaluation** of generative and non-generative queries
- Insights into **tool usage effectiveness**
- Trend analysis on **reasoning depth** and **difficulty**
- Exportable CSVs for further aggregation or leaderboard integration
