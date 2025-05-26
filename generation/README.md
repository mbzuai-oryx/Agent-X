# Query and Reasoning Generation for Agent-X

This folder contains the **generation pipeline** for the Agent-X benchmark. It includes scripts for:

1. Generating natural language queries based on multimodal inputs (images or video frames)
2. Generating reasoning traces using refined queries and a defined toolset

These outputs are used to construct agentic reasoning tasks in the benchmark.

---

## Files

- `query_generation.py` – Generates tool-use queries from images using GPT-4o.
- `reasoning_generation.py` – Generates structured reasoning traces for each refined query.
- `toolmeta.json` – JSON metadata describing each available tool’s inputs, outputs, and descriptions.

---

## Prerequisites

- Python 3.8+
- Required packages:

```bash
pip install openai pandas
```

- `openai_key.txt` – A text file containing your OpenAI API key

---

## 1. Query Generation

**Script:** `query_generation.py`  
**Purpose:** Generate high-quality, tool-relevant multimodal queries from input images using GPT-4o.

### Inputs

- A folder of `.jpg`, `.jpeg`, or `.png` images located at `./reasoning/Generative/`
- OpenAI key in `openai_key.txt`

### Outputs

- `generative/Generative_Queries.xlsx` – Excel file containing:
  - File path
  - Generated query
  - (Blank) columns for human-refined query and type
- `generative/failed_images_b5.txt` – Images that failed during processing
- The script also cleans image paths for portability.

---

## 2. Human Refinement Step (Manual)

After running `query_generation.py`, open the Excel file and:

- Refine each query to be clearer, realistic, and human-evaluable
- Fill in the "Final Query" and "Final Query Type" columns
- Save it as:

```bash
B1/refined_queries_B1.xlsx
```

---

## 3. Reasoning Generation

**Script:** `reasoning_generation.py`  
**Purpose:** Generate detailed reasoning traces from refined queries using GPT-4o and the provided tools.

### Inputs

- `B1/refined_queries_B1.xlsx` – File with finalized queries
- `toolmeta.json` – Describes available tools and their I/O
- `./reasoning/B1/` – Folder with:
  - Images for single-image queries
  - Subfolders or frame extractions for video queries

### Outputs

- `B1/generated_agent_reasoning_B1.xlsx` – Excel file containing:
  - Image path
  - Final query and type
  - Structured reasoning steps (tool + input/output + thought)
  - Final answer and justification
  - Step and tool counts

---

## Output Format

Each reasoning trace is saved as a structured JSON-like object:

```json
[
  {
    "step": 1,
    "task": "...",
    "tool": "...",
    "input": "...",
    "output": "...",
    "thought": "..."
  },
  ...
  {
    "final_answer": {
      "value": "...",
      "justification": "..."
    }
  }
]
```

---

## Notes

- `TEST_FILTER` in `reasoning_generation.py` can be used to run a subset of rows for debugging.
- Video-based queries rely on extracted frames in `reasoning/B1/video_frames/`.
- Ensure `toolmeta.json` follows the correct schema: tool name, inputs, outputs, and description.

---

## Example Directory Structure

```
generation/
├── query_generation.py
├── reasoning_generation.py
├── toolmeta.json
├── openai_key.txt
├── generative/
│   └── Generative_Queries.xlsx
├── reasoning/
│   └── B1/
│       ├── refined_queries_B1.xlsx
│       ├── generated_agent_reasoning_B1.xlsx
│       └── video_frames/
```

---

## Citation

These scripts are part of the **Agent-X** benchmark for evaluating vision-language agents on tool-augmented reasoning tasks.

---

## Contact

For questions, please reach out to the authors listed in the main Agent-X paper.
