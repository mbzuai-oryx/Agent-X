# 🧠 Agent-X: Generation Pipeline

This directory contains scripts used to generate the **query–reasoning pairs** that form the basis of the Agent-X benchmark. The pipeline consists of three main stages:

- `video_frames.py`: for extracting key frames from videos
- `query_generation.py`: for generating realistic, multi-step queries from images and video frames
- `reasoning_generation.py`: for producing complete step-by-step tool-use reasoning traces in response to the human refined queries

---

## 🎬 0. Video Frame Extraction

**Script:** `video_frames.py`

This utility dynamically extracts frames from video files to use as input for the Agent-X pipeline. Since most vision-language models do not accept raw video input, we sample frames instead.

### 🔹 What It Does

- Reads all `.mp4`, `.avi`, `.mov`, or `.mkv` files in a given folder
- Dynamically selects `nth` frame based on video length:
  - ≤50 frames → every 3rd frame
  - ≤200 frames → every 10th frame
  - ≤500 frames → every 20th frame
  - ≤1000 frames → every 30th frame
  - >1000 frames → every 50th frame
- Saves extracted frames to an output folder as `.jpg`

### 🔹 Usage

Call the function `extract_every_nth_frame_dynamic(video_folder, output_folder)` in your script or notebook. The output can then be used in `query_generation.py`.

---

## ✨ 1. Query Generation

**Script:** `query_generation.py`

This script uses OpenAI's GPT-4o to generate **natural language queries** based on visual inputs. The queries are constrained to be complex, tool-relevant, and human-verifiable, matching Agent-X task requirements.

### 🔹 Input

- A folder of `.jpg`, `.jpeg`, or `.png` files (configurable via `folder_path`)

### 🔹 Output

- `Generative_Queries.xlsx`:
  - `File path`
  - `Generated Query`
  - (empty columns for human-refined queries and types)
- `failed_images.txt` — list of files that failed to process
- Optional: cleaned file paths in the Excel sheet

### 📋 Prompt Constraints

The model is instructed to:

- Require at least **3 reasoning steps**
- Use at least **2 tools** (from a predefined list)
- Not mention tool names directly
- Stay grounded in **realistic**, **time-invariant**, and **evaluable** queries

---

## 🔎 2. Reasoning Generation

**Script:** `reasoning_generation.py`

This script takes human refined queries and generates detailed **reasoning traces**, including the tools used, their input/output, and the thought process for each step.

### 🔹 Input

- Refined queries from `refined_queries_<batch>.xlsx` (editable by annotators)
- Image(s) or video frame folder corresponding to each query
- Tool metadata from `toolmeta.json`

### 🔹 Output

- `generated_agent_reasoning_<batch>.xlsx`:
  - `Query`
  - `Reasoning Steps` (tool/task/thought trace)
  - `Final Answer` and its `Justification`
  - `Tools Used` and `Total Steps`

### 🧠 What It Does

- Builds a system prompt with all tool descriptions
- Encodes image(s) or video frames to base64
- Feeds the query + image(s) to GPT-4o
- Parses the response into structured JSON and saves the results

---

## 🧪 Notes

- Both scripts require an OpenAI API key in a file named `openai_key.txt`
- The generation pipeline is semi-automated: queries are **machine-generated then human-refined**, and final reasoning is again **machine-generated then validated**
- Intermediate outputs are stored in Excel format to simplify human annotation
- The final dataset is exported in JSON format for integration with the Agent-X benchmark

---

## 🔧 Configuration Tips

- Change `folder_path`, `BATCH`, or output filenames at the top of each script
- Ensure `toolmeta.json` is correctly formatted with tool names, inputs, and output descriptions
