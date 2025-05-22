import requests
import os
import io
from PIL import Image
from urllib.request import urlopen
import os
import json
import time
import re
import cv2
import json
import shutil
from tempfile import TemporaryDirectory
import cv2
import random
import requests
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

from typing import Union

import re
import json
from typing import Union,Optional
from typing import Optional, Dict, Any

def extract_last_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the last valid JSON dictionary that starts with 'query' and ends with 'final_answer' from a given string.

    Parameters:
        text (str): The input text containing one or more JSON-like objects.

    Returns:
        dict: The last valid parsed JSON object, or None if not found or invalid.
    """
    # Regex to match any JSON object that starts with "query" and ends with "final_answer": { ... }
    pattern = r'(\{\s*"query"\s*:\s*.*?"final_answer"\s*:\s*\{.*?\}\s*\})'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    last_json_str = matches[-1]

    try:
        return json.loads(last_json_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to decode JSON: {e}")
        return None






def extract_sampled_frames(video_path, num_samples=4):
    # Create a temporary directory to store extracted frames
    temp_dir = TemporaryDirectory()
    frame_paths = []

    # Open the video file using OpenCV
    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    
    # Randomly select frames to extract
    frame_indices = sorted(random.sample(range(total_frames), num_samples))

    for idx in frame_indices:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video_cap.read()
        if ret:
            frame_filename = os.path.join(temp_dir.name, f"frame_{idx}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
    
    video_cap.release()
    
    # Return the list of frame file paths
    return frame_paths, temp_dir

tool_metadata_path = "/share/data/drive_1/hanan/multiagent_eval_data/toolmeta.json"
with open(tool_metadata_path, "r") as f:
    meta_data = json.load(f)
f.close()
meta_data = json.dumps(meta_data)

p = """
    - "reasoning_step_format": [
        {"task": "Describe the sub-task being performed."},
        {"tool_used": "Specify the tool selected and justify its choice."},
        {"tool_output": "Provide the tool's output."},
        {"thought": "Explain the significance of the output and how it contributes to answering the query."}
    ]

    - "constraints": [
        "Use tools only when necessary and justify their usage.",
        "Ensure each step is self-contained and clearly explained.",
        "Maintain transparency in decision-making and reasoning."
    ]

    - "final_answer": "Provide a clear and concise answer based on all previous steps", "justification": "Provide a justification for the final answer."    
    }

    "The output must be a single dictionary containing query, reasoning steps and final answer as shown in the following example":\n

        { "query": "What is the man doing in the video?",
        "reasoning_steps": [
            {
                "step": 1,
                "task": "Identify key frames containing activity.",
                "tool": "Scene Segmentation Tool",
                "tool_output": "Frames 40-80 show continuous motion.",
                "thought": "These frames likely contain the main action."
            },
            {
                "step": 2,
                "task": "Recognize the action happening in the selected frames.",
                "tool_used": "Action Recognition Model",
                "tool_output": "Man is playing guitar",
                "thought": "The model confirms that the man is engaged in a musical activity."
            }
        ],
        "final_answer": {"value": "The man is playing a guitar in the video.", "justification":  "Through scene segmentation and targeted action recognition, we isolated the most relevant frames and identified the man's activity with high confidence, leading to a precise and grounded answer."
            }
        }
"""
instruction_prompt = "You are an intelligent multi-modal assistant. You are provided with:\n" \
             "- A text query\n" \
             "- An image or video\n" \
             "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"\
             f"{meta_data}\n\n" \
             "Your objective is to answer the query based on the given visual content " \
             "by choosing and using the most appropriate tools. You must reason step-by-step. " \
             "Each reasoning step should include: \n" \
             + p

instruction_prompt = f"""
You are an intelligent multi-modal assistant.

You are provided with:
- A text query
- An image or video
- A set of tools to assist with your reasoning, whose metadata is provided below:

{meta_data}

---

## 🧠 Task:
Answer the query by reasoning step-by-step and using tools only when necessary. Use structured reasoning and justify your choices clearly.

---

## ⚠️ Output Format (STRICTLY REQUIRED):

⚠️ Your **entire response must be ONLY a single valid JSON object**, structured as follows:

```json
{{
  "query": "<The original query>",
  "reasoning_steps": [
    {{
      "step": 1,
      "task": "Describe the sub-task being performed.",
      "tool": "Specify the tool selected and justify its choice.",
      "tool_output": "Provide the tool's output.",
      "thought": "Explain how this output contributes to answering the query."
    }},
    ...
  ],
  "final_answer": {{
    "value": "<Concise answer to the original query>",
    "justification": "<Justify the answer based on prior reasoning steps>"
  }}
}}
"""


reasoning_data_path ="/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/metadata/verified_data.json"
with open(reasoning_data_path, "r") as g:
    reason_data = json.load(g)


## model inference

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# model_id = "Llama-3.2-11B-Vision"

# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(model_id)



base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"
save_path  = "llama32_results_final.json"
required_keys = ["reasoning_steps", "final_answer"]
final_results = []
max_attempts = 3

for key, value in reason_data.items():
    d = {}
# try:
    video_flag = False
    data = reason_data[key][0]
    sample = data["file_path"]
    sample_list = [img.strip() for img in sample.split(",")]
    if len(sample_list)>1 :
        continue
    if sample_list[0].split(".")[1].lower() in ["mp4", "avi", "mov"]:
        video_flag = True
        continue
    sample_path = [os.path.join(base_path, s) for s in sample_list]
    query = data["query"]

    valid_response = False

    for attempt in range(max_attempts):
        try:
            # Build messages
            if not video_flag:
                if len(sample_path) == 1:
                    
                    chat = [{"role": "syste,","content": instruction_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text":  "Query: " + query},
                            ]
                        }
                    ]
                else:
                    chat = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "image"},
                                {"type": "text", "text":instruction_prompt + "\n" + "Query: " + query},
                            ]
                        },

                    ] 
            else:
                frame_paths, temp_dir = extract_sampled_frames(sample_path[0], num_samples=4)
                chat = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": instruction_prompt + "\n" + "Query: " + query},
                        ]
                    },
                ]
                sample_path = frame_paths[:1]
            # Generate
            image = [Image.open(im) for im in sample_path]
            input_text = processor.apply_chat_template(chat, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            output = model.generate(**inputs, max_new_tokens=1024)
            response = processor.decode(output[0])
            #output_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output_text.strip(), flags=re.IGNORECASE)
            output_text = extract_last_json_block(response)

            print(f"Attempt {attempt + 1} response for key {key}:\n{response}")

            if output_text:
                try:
                    response_dict = output_text  #json.loads(output_text)
                    
                    if all(k in response_dict for k in required_keys):
                        d[key] = {
                            "query": query,
                            "filename": sample,
                            **response_dict
                        }
                        valid_response = True
                        break  # success
                except json.JSONDecodeError:
                    print(f"Invalid JSON on attempt {attempt + 1} for key {key}: {output_text}")
                    pass

            time.sleep(5)

        except Exception as attempt_err:
            print(f"Exception on attempt {attempt + 1} for key {key}: {attempt_err}")
            time.sleep(5)
            continue

    if not valid_response:
        d[key] = {
            "query": query,
            "filename": sample,
            "reasoning_steps": "",
            "final_answer": {
                "value": "",
                "justification": ""
            }
        }

    final_results.append(d)
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    # if video_flag:
    #     shutil.rmtree(temp_dir.name)

    # except Exception as e:
    #     print(f"Exception for key {key}: {e}")
    #     continue




with open('llama32_results_final_backup.json', 'w') as f:
    json.dump(final_results, f)


