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
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import re
import json
import argparse
parser = argparse.ArgumentParser()

def extract_last_json(text: str):
    """
    Extracts the last valid JSON object from a text string using balanced brace matching.
    Returns the last successfully parsed JSON object, or None if none found.
    """
    stack = []
    start_indices = []
    candidates = []

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_indices.append(i)
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_indices:
                    start = start_indices.pop()
                    candidate = text[start:i+1]
                    candidates.append(candidate)

    # Try to parse JSON from the end (last candidate first)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

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


instruction_prompt = """
You are an intelligent multi-modal assistant. You are provided with:
- A text query
- An image or video
- A set of tools to assist with your reasoning (see tool metadata below)

{meta_data}

Your objective is to answer the query by analyzing the visual content using appropriate tools.
You must reason step-by-step, using tools only when necessary and justifying their use.

### Reasoning Instructions:
Each reasoning step must include the following fields:
- "step": Integer step number (starting from 1)
- "task": Describe the sub-task being performed.
- "tool" or "tool_used": Specify the tool selected and justify its choice.
- "tool_output": Provide the tool's output.
- "thought": Explain the significance of the output and how it contributes to answering the query.

### Constraints:
- Only use tools when necessary and always justify their use.
- Ensure each reasoning step is self-contained and clearly explained.
- Maintain transparency in your decision-making and reasoning.

### Final Output Format:
Your entire response must be a **single JSON dictionary** containing only the following three keys:
- "query": A string representing the original input query.
- "reasoning_steps": A list of dictionaries, one for each step of your reasoning, as described above.
- "final_answer": A dictionary containing:
    - "value": The final answer to the query.
    - "justification": A brief explanation of how the reasoning steps support this answer.

### Example Output:
```json
{{
  "query": "What is the man doing in the video?",
  "reasoning_steps": [
    {{
      "step": 1,
      "task": "Identify key frames containing activity.",
      "tool": "Scene Segmentation Tool",
      "tool_output": "Frames 40-80 show continuous motion.",
      "thought": "These frames likely contain the main action."
    }},
    {{
      "step": 2,
      "task": "Recognize the action happening in the selected frames.",
      "tool_used": "Action Recognition Model",
      "tool_output": "Man is playing guitar",
      "thought": "The model confirms that the man is engaged in a musical activity."
    }}
  ],
  "final_answer": {{
    "value": "The man is playing a guitar in the video.",
    "justification": "Through scene segmentation and targeted action recognition, we isolated the most relevant frames and identified the man's activity with high confidence."
  }}
}}
"""


## model load
model_path = "moonshotai/Kimi-VL-A3B-Thinking"
model = AutoModelForCausalLM.from_pretrained(
     model_path,
     torch_dtype=torch.bfloat16,
     device_map="auto",
     trust_remote_code=True,
     attn_implementation="flash_attention_2"
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
required_keys = ["reasoning_steps", "final_answer"]


if __name__=="__main__":

    final_results = []
    max_attempts = 2

    # Adding arguments
    parser.add_argument("--save_path",default="kimivl_final_results.json", help = "path to Output file")
    parser.add_argument("--base_path", default="./AgentX/files",help = "path to data folder")
    parser.add_argument("--tool_data_path", default="./AgentX/tools_metadata.json", help = "path to tool metadata json file")
    parser.add_argument("--gt_data_path", default="./AgentX/data.json", help = "path to ground truth json file")

    # Read arguments from command line
    args = parser.parse_args()

        ## read the gt reason data
    with open(args.gt_data_path, "r") as g:
        reason_data = json.load(g)
    g.close()
    
    ## read the tool meta data
    with open(args.tool_data_path, "r") as f:
        meta_data = json.load(f)
    f.close()
    meta_data = json.dumps(meta_data)
    instruction_prompt = instruction_prompt.format(meta_data=meta_data)

    for key, value in reason_data.items():
        d = {}
        
        try:
            video_flag = False
            data = reason_data[key][0]
            sample = data["file_path"]
            sample_list = [img.strip() for img in sample.split(",")]
            if sample_list[0].split(".")[1].lower() in ["mp4", "avi", "mov"]:
                video_flag = True
            sample_path = [os.path.join(args.base_path, s) for s in sample_list]
            query = data["query"]

            valid_response = False

            for attempt in range(max_attempts):
                try:
                    # Build messages
                    if not video_flag:
                        if len(sample_path) == 1:
                            chat = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": sample_path[0]},
                                        {"type": "text", "text": instruction_prompt + "\n" + "Query: " + query},
                                    ]
                                }
                            ]
                        else:
                            chat = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": sample_path[0]},
                                        {"type": "image", "image": sample_path[1]},
                                        {"type": "text", "text": instruction_prompt + "\n" + "Query: " + query}],
                                    
                                },

                            ]
                    else:
                        frame_paths, temp_dir = extract_sampled_frames(sample_path[0], num_samples=4)
                        chat = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": frame_paths[0]},
                                    {"type": "image", "image": frame_paths[1]},
                                    {"type": "image", "image": frame_paths[2]},
                                    {"type": "text", "text": instruction_prompt + "\n" + "Query: " + query},
                                ]
                            },
                        ]
                        sample_path = frame_paths[:3]

                    # Generate
                    images = [Image.open(path) for path in sample_path]
                    text = processor.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt")
                    inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
                    generated_ids = model.generate(**inputs, max_new_tokens=2048)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    #output_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
                    output_text = extract_last_json(response)

                    print(f"Attempt {attempt + 1} response for key {key}:\n{response}")

                    if response:
                        try:
                            response_dict = output_text   #json.loads(output_text)
                            
                            if all(k in response_dict for k in required_keys):
                                d[key] = {
                                    "query": query,
                                    "filename": sample,
                                    **response_dict
                                }
                                valid_response = True
                                break  # success
                        except json.JSONDecodeError:
                            print(f"Invalid JSON on attempt {attempt + 1} for key {key}: {response}")
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
            with open(args.save_path, "w") as f:
                json.dump(final_results, f, indent=2)
            
            if video_flag:
                shutil.rmtree(temp_dir.name)

        except Exception as e:
            print(f"Exception for key {key}: {e}")
            continue

