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
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu    # pip install decord


MAX_NUM_FRAMES=4
def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def extract_last_json_block(text: str) -> dict:
    """
    Extracts the last JSON dictionary enclosed in ```json ... ``` from a given text string.
    
    Args:
        text (str): The input text containing JSON blocks.
    
    Returns:
        dict: The parsed JSON dictionary from the last block, or None if no valid block is found.
    """
    matches = re.findall(r'```json\s*({.*?})\s*```', text, re.DOTALL)
    
    if matches:
        last_json_str = matches[-1]
        try:
            return json.loads(last_json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON block found.")
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
instruction_prompt = instruction_prompt.format(meta_data=meta_data)

reasoning_data_path ="/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/metadata/verified_data.json"
with open(reasoning_data_path, "r") as g:
    reason_data = json.load(g)


## model inference

model_path = '/share/data/drive_2/hanan/mPLUG-Owl/mPLUG-Owl3/mPLUG-Owl3-7B-240728'
config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
print(config)
# model = mPLUGOwl3Model(config).cuda().half()
model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.half,trust_remote_code=True)
model.eval().cuda()
model_path = '/share/data/drive_2/hanan/mPLUG-Owl/mPLUG-Owl3/mPLUG-Owl3-7B-240728'
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)



base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"
save_path  = "mplugowl_results_final.json"
required_keys = ["reasoning_steps", "final_answer"]
final_results = []
max_attempts = 1

for key, value in reason_data.items():
    d = {}
# # try:
#     if int(key)<=432:
#         continue
    video_flag = False
    data = reason_data[key][0]
    sample = data["file_path"]
    sample_list = [img.strip() for img in sample.split(",")]
    if sample_list[0].split(".")[1].lower() in ["mp4", "avi", "mov"]:
        video_flag = True
    sample_path = [os.path.join(base_path, s) for s in sample_list]
    query = data["query"]

    valid_response = False

    for attempt in range(max_attempts):
        # try:
            # Build messages
            if not video_flag:
                if len(sample_path) == 1:
                    messages = [
                        {"role": "user", "content": "<|image|>" + "\n" + instruction_prompt + "\n" + "Query: " + query},
                        {"role": "assistant", "content": ""}
                    ]

                else:
                    messages = [
                        {"role": "user", "content": "<|image|><|image|>" + "\n" + instruction_prompt + "\n" + "Query: " + query},
                        {"role": "assistant", "content": ""}
                    ]
                image = [Image.open(p).convert('RGB') for p in sample_path]
                inputs = processor(messages, images=image, videos=None)

                inputs.to('cuda')
                inputs.update({
                    'tokenizer': tokenizer,
                    'max_new_tokens':1024,
                    'decode_text':True,
                })


                response = model.generate(**inputs)[0]
            else:
                frames = encode_video(sample_path[0])
                messages = [
                        {"role": "user", "content": "<|video|>" + "\n" + instruction_prompt + "\n" + "Query: " + query},
                        {"role": "assistant", "content": ""}
                    ]

                inputs = processor(messages, images=None, videos=[frames])

                inputs.to('cuda')
                inputs.update({
                    'tokenizer': tokenizer,
                    'max_new_tokens':1024,
                    'decode_text':True,
                })

                response = model.generate(**inputs)[0]

            
            #output_text = extract_full_json_with_final_answer(output)
            output_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
            #output_text = extract_last_json_block(output)

            print(f"Attempt {attempt + 1} response for key {key}:\n{response}")

            if output_text:
                try:
                    response_dict = json.loads(output_text)
                    
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

        # except Exception as attempt_err:
        #     print(f"Exception on attempt {attempt + 1} for key {key}: {attempt_err}")
        #     time.sleep(5)
        #     continue

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




with open('mplugowl_results_final_backup.json', 'w') as f:
    json.dump(final_results, f)


