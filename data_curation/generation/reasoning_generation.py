import os
import json
import base64
import pandas as pd
from openai import OpenAI
from openai._exceptions import OpenAIError, APIError

BATCH = "B1"
input_xlsx = f"{BATCH}/refined_queries_{BATCH}.xlsx"
toolmeta_file = "toolmeta.json"
output_xlsx = f"{BATCH}/generated_agent_reasoning_{BATCH}.xlsx"
base_folder = f"./reasoning"
frames_folder = f"./reasoning/{BATCH}/video_frames"

with open("openai_key.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

with open(toolmeta_file) as f:
    toolmeta = json.load(f)

def build_tool_usage_section(toolmeta):
    lines = []
    for tool in toolmeta.values():
        lines.append(f"{tool['name']}:")
        lines.append(f"- Description: {tool.get('description', '').strip()}")
        for inp in tool.get("inputs", []):
            if inp["type"] == "image":
                lines.append("- Input: An image file")
            elif inp["type"] == "text":
                lines.append("- Input: Text input")
        for out in tool.get("outputs", []):
            lines.append(f"- Output: {out.get('description', 'Text result')}")
        lines.append("")
    return "\n".join(lines)

def build_reasoning_prompt(query, image_path, toolmeta_section):
    return f"""
You are an agent performing multimodal reasoning using the tools listed below.

Your goal is to answer the given query step-by-step by selecting the right tools for each stage of reasoning. 
For each step:
- Clearly state what you're trying to do.
- Specify the tool you are using.
- Explicitly include:
  - The input provided to the tool
  - The output received from the tool
- Explain what you learned or why this step was necessary.

Make sure your tool usage follows the tool descriptions provided. Do not hallucinate tools or skip intermediate reasoning steps.

---

Available Tools & How to Use Them:

{toolmeta_section}

---

Query:
"{query}"

Images:
{', '.join([img[0] for img in image_blobs])}


---

Return your output as a JSON list of step, formatted like this:

[
  {{
    "step": <step number>,
    "task": "<Short description of what this step is trying to do>",
    "tool": "<Tool name(s)>",
    "input": "<Input provided to the tool>",
    "output": "<Output from the tool>",
    "thought": "<Why this step was needed, or what you learned>"
  }},
  ...
  {{
    "final_answer": {{
      "value": "<Final conclusion>",
      "justification": "<How all steps led to the answer>"
    }}
  }}
]""".strip()

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


df = pd.read_excel(input_xlsx)
df.columns = df.columns.str.strip()
samples = df[df['Final Query'].notna()].to_dict(orient="records")



TEST_FILTER = ""  

if TEST_FILTER:
    before_count = len(samples)
    samples = [s for s in samples if s['File path'].endswith(TEST_FILTER)]

    after_count = len(samples)
    print(f"Test filter active: only processing rows containing '{TEST_FILTER}'")
    print(f"Matched {after_count} out of {before_count} rows.")
else:
    print(f"No test filter set. Processing all {len(samples)} rows.")

rows = []
toolmeta_section = build_tool_usage_section(toolmeta)

for sample in samples:
    relative_path = sample['File path']
    query = sample['Final Query']
    query_type = sample.get('Final Query Type', '')

    print(f"\n🔍 Processing: {relative_path}")

    image_blobs = []


    # Case 1: Single image
    if relative_path.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(base_folder, relative_path)
        if os.path.exists(full_path):
            image_blobs = [(relative_path, encode_image(full_path))]
        else:
            print(f"File not found: {relative_path}")
            continue

    # Case 2: Folder of images
    elif os.path.isdir(os.path.join(base_folder, relative_path)):
        folder_path = os.path.join(base_folder, relative_path)
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_img_path = os.path.join(folder_path, fname)
                rel = os.path.join(relative_path, fname)
                image_blobs.append((rel, encode_image(full_img_path)))
        if not image_blobs:
            print(f"Folder found but no images inside: {relative_path}")
            continue

    # Case 3: Video
    else:
        frame_prefix = os.path.splitext(os.path.basename(relative_path))[0]
        frame_matches = sorted([
            f for f in os.listdir(frames_folder)
            if f.startswith(frame_prefix) and f.endswith(".jpg")
        ])
        for f in frame_matches:
            frame_path = os.path.join(frames_folder, f)
            rel = os.path.join(f"{BATCH}/video_frames", f)
            image_blobs.append((rel, encode_image(frame_path)))
        if not image_blobs:
            print(f"No matching video frames found for: {relative_path}")
            continue

    
    prompt = build_reasoning_prompt(query, relative_path, toolmeta_section)

    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for rel_path, img_b64 in image_blobs:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        print("Sending to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        content = response.choices[0].message.content.strip()
        #print("\nGPT-4o Response:\n", content)

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(content)

        steps = []
        tools = set()
        tool_io_lines = []
        final_value = "N/A"
        justification = ""

        for step in parsed:
            if "final_answer" in step:
                final_value = step["final_answer"].get("value", "")
                justification = step["final_answer"].get("justification", "")
            else:
                steps.append(
                    f"Step: {step['step']}\nTask: {step['task']}\nTool: {step['tool']}\nThought: {step['thought']}"
                )
                tools.add(step['tool'])
                #tool_io_lines.append(f"{step['tool']} (input: {step['input']}, output: {step['output']})")
                # Use actual image paths as input if available
                used_images = ", ".join([img[0] for img in image_blobs])
                step_input = step.get("input", "")
                if "{image}" in step_input:
                    step_input = step_input.replace("{image}", used_images)

                tool_io_lines.append(f"{step['tool']} (input: {step_input}, output: {step['output']})")

        rows.append([
            relative_path,
            query,
            query_type,
            "\n\n".join(steps),
            len(steps),
            "\n\n".join(tool_io_lines),
            final_value,
            justification
        ])

        print(f"\nReasoning completed for {relative_path}")

    except Exception as e:
        print(f"Failed for {relative_path}: {e}")

if rows:
    df_out = pd.DataFrame(rows, columns=[
        "Image path", "Query", "Query Type", "Reasoning Steps (thoughts only)",
        "Total Steps", "Tools Used (input/output)", "Final Answer", "Justification"
    ])
    df_out.to_excel(output_xlsx, index=False)
    print(f"\nSaved reasoning to {output_xlsx}")
else:
    print("No samples successfully processed.")
