import base64
import openai
from openai import OpenAI
import json
import os
import pandas as pd

# === CONFIG ===
with open("openai_key.txt", "r") as keyfile:
    api_key = keyfile.read().strip()

client = OpenAI(api_key=api_key)
folder_path = "./reasoning/Generative"
output_xlsx = "generative/Generative_Queries.xlsx"
failed_images_file = "generative/failed_images_b5.txt"

rows = []
failed_images = []

# Define available tools
tools = [
    "ImageDescription", "CountGivenObject", "OCR", "DrawBox", "Calculator",
    "DetectGivenObject", "RegionAttributeDescription", "MathOCR", "Solver",
    "Plot", "GoogleSearch", "TextToImage", "AddText", "ImageStylization"
]

def build_annotation_prompt(input_files, tool_list):
    file_repr = json.dumps(input_files)
    return f"""
"You are an annotator tasked with generating a realistic and verifiable user
query for benchmarking a multimodal LMM-based assistant.
You are provided with the following tools the assistant can use:
tools = [ "ImageDescription", "CountGivenObject", "OCR", "DrawBox",
"Calculator", "DetectGivenObject", "RegionAttributeDescription",
"MathOCR", "Solver", "Plot", "GoogleSearch", "TextToImage", "AddText",
"ImageStylization" ]
Your job is to generate one single complex and human-evaluable user query
that satisfies all of the following:
• If you are given multiple images, the query must require reasoning
across multiple images (not just one)
• If you are given multiple frames, the query must involve comparing,
tracking, or analyzing across multiple video frames.
• The query must require at least 3 distinct reasoning steps to be
answered.
• It must require at least 2 different tools from the list above.
• Do not mention tool names explicitly.
• If the query involves online or time-sensitive content, include a
fixed timeframe or source.
• The query must be realistic and grounded in common user intentions.
• Use only English.
• All queries must be suitable for evaluation by a human — the answer
should not vary arbitrarily across individuals.
Return your result as a single JSON object on one line.
{{ "input": {file_repr}, "query": "<natural language query>" }}"
"""

# Get all image files
image_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Process images
for image_path in sorted(image_files):
    print(f"🔍 Processing {image_path}...")

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Could not read image {image_path}: {e}")
        failed_images.append(image_path)
        continue

    prompt = build_annotation_prompt([image_path], tools)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        result_line = response.choices[0].message.content.strip()

        if not result_line:
            print(f"⚠️ Empty response for {image_path}")
            failed_images.append(image_path)
            continue

        if result_line.startswith("```json"):
            result_line = result_line.replace("```json", "").replace("```", "").strip()

        query_obj = json.loads(result_line)

        rows.append([
            image_path,
            query_obj.get("query", ""),
            "",  # Final query
            ""   # Final query type
        ])

        print("✅ Success")

    except Exception as e:
        print(f"❌ Failed on {image_path}: {e}")
        failed_images.append(image_path)
        continue

# Save everything to Excel
df = pd.DataFrame(rows, columns=[
    "File path", "Generated Query", "Final Query", "Final Query Type"
])
df.to_excel(output_xlsx, index=False)
print(f"\n✅ XLSX export completed: {output_xlsx}")

# Save failed images to a text file
if failed_images:
    with open(failed_images_file, "w") as f:
        for path in failed_images:
            f.write(path + "\n")
    print(f"⚠️ Saved {len(failed_images)} failed images to {failed_images_file}")
else:
    print("🎯 No failed images. Everything succeeded.")


def clean_image_paths_inplace(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df['File path'] = df['File path'].apply(lambda x: os.path.join(*x.split("/")[-2:]) if isinstance(x, str) else x)
    df.to_excel(xlsx_path, index=False)
    print(f"✅ Overwritten {xlsx_path} with cleaned paths.")

clean_image_paths_inplace(output_xlsx)

