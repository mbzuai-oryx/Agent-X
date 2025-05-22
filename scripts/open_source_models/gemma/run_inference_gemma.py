import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import AutoTokenizer, Gemma3ForCausalLM
import os
import json
import time
import re
import shutil
from tempfile import TemporaryDirectory
import cv2
import random

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

    - "final_output": {
        "final_answer": "Provide a clear and concise answer based on all previous steps", "justification": "Provide a justification for the final answer."    
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

reasoning_data_path ="/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/metadata/verified_data.json"
with open(reasoning_data_path, "r") as g:
    reason_data = json.load(g)


os.environ['HF_TOKEN'] = "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"

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    ckpt, device_map="auto", torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(ckpt)


base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"
save_path  = "gemma_results_final.json"
required_keys = ["reasoning_steps", "final_answer"]
final_results = []
max_attempts = 5

for key, value in reason_data.items():
    d = {}
    try:
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
            try:
                # Build messages
                if not video_flag:
                    if len(sample_path) == 1:
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": instruction_prompt}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": sample_path[0]},
                                    {"type": "text", "text": "Query: " + query},
                                ]
                            },

                        ]
                    else:
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": instruction_prompt}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": sample_path[0]},
                                    {"type": "image", "image": sample_path[1]},
                                    {"type": "text", "text": "Query: " + query},
                                ]
                            },

                        ]
                else:
                    frame_paths, temp_dir = extract_sampled_frames(sample_path[0], num_samples=4)
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": instruction_prompt}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": frame_paths[0]},
                                {"type": "image", "image": frame_paths[1]},
                                {"type": "image", "image": frame_paths[2]},
                                {"type": "image", "image": frame_paths[3]},
                                {"type": "text", "text": "Query: " + query},
                            ]
                        },
                    ]

                # Generate
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                generation = generation[0][input_len:]

                decoded = processor.decode(generation, skip_special_tokens=True)
                output_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", decoded.strip(), flags=re.IGNORECASE)

                print(f"Attempt {attempt + 1} response for key {key}:\n{output_text}")

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
        
        if video_flag:
            shutil.rmtree(temp_dir.name)

    except Exception as e:
        print(f"Exception for key {key}: {e}")
        continue




with open('gemma_results_final_backup.json', 'w') as f:
    json.dump(final_results, f)
