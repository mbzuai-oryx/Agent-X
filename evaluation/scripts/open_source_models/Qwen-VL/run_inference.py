import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
import time
import re
import argparse
parser = argparse.ArgumentParser()


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


device = "cuda:0"
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path)


required_keys = ["reasoning_steps", "final_answer"]

if __name__ =="__main__":


    final_results = []
    max_attempts = 2

    # Adding arguments
    parser.add_argument("--save_path",default="qwen_final_results.json", help = "path to Output file")
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

    instruction_prompt = "You are an intelligent multi-modal assistant. You are provided with:\n" \
             "- A text query\n" \
             "- An image or video\n" \
             "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"\
             f"{meta_data}\n\n" \
             "Your objective is to answer the query based on the given visual content " \
             "by choosing and using the most appropriate tools. You must reason step-by-step. " \
             "Each reasoning step should include: \n" \
             + p

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
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": sample_path[0]},
                                        {"type": "text", "text": instruction_prompt + "\n\n Query: " + query},
                                    ],
                                }
                            ]
                        else:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": sample_path[0]},
                                        {"type": "image", "image": sample_path[1]},
                                        {"type": "text", "text": instruction_prompt + "\n\n Query: " + query},
                                    ],
                                }
                            ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)

                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to("cuda")
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "video", "video": sample_path[0], "fps": 1.0},
                                    {"type": "text", "text": instruction_prompt + "\n\nQuery: " + query},
                                ],
                            }
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                            **video_kwargs,
                        ).to("cuda")

                    # Generate
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    output_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", output_text.strip(), flags=re.IGNORECASE)

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
            with open(args.save_path, "w") as f:
                json.dump(final_results, f, indent=2)

        except Exception as e:
            print(f"Exception for key {key}: {e}")
            continue
