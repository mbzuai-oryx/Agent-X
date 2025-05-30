import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import json
import time
import argparse
parser = argparse.ArgumentParser()

p = {
    "reasoning_step_format": [
        {"task": "Describe the sub-task being performed."},
        {"tool_used": "Specify the tool selected and justify its choice."},
        {"tool_output": "Provide the tool's output."},
        {"thought": "Explain the significance of the output and how it contributes to answering the query."}
    ],
    "constraints": [
        "Use tools only when necessary and justify their usage.",
        "Ensure each step is self-contained and clearly explained.",
        "Maintain transparency in decision-making and reasoning."
    ],
    "final_output": {
        "final_answer": "Provide a clear and concise answer based on all previous steps", "justification": "Provide a justification for the final answer."
    },
    "The output should be a python dictionary as shown in the following example":
    # "example": {
        {
        "query": "What is the man doing in the video?",
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
}



## load model

device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B-Image"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


required_keys = ["reasoning_steps", "final_answer"]

if __name__ =="__main__":


    final_results = []
    max_attempts = 2

    # Adding arguments
    parser.add_argument("--save_path",default="videollama3_final_results.json", help = "path to Output file")
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
    instruction_prompt = "You are an intelligent multi-modal agent. You are provided with:\n" \
             "- A text query\n" \
             "- An image or video\n" \
             "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"\
             f"{meta_data}\n\n" \
             "Your objective is to answer the query based on the given visual content " \
             "by choosing and using the most appropriate tools. You must reason step-by-step. " \
             "Each reasoning step should include: \n\n" \
              + json.dumps(p, indent=2)

    for key, value in reason_data.items():
        d = {}
        try:
            video_flag = False
            data = reason_data[key][0]
            sample = data["file_path"]
            sample_list = [img.strip() for img in sample.split(",")]
            if sample_list[0].split(".")[1] in ["mp4", "avi", "mov"]:
                video_flag = True
            sample_path = [os.path.join(args.base_path, s) for s in sample_list]
            query = data["query"]

            valid_response = False
            for attempt in range(max_attempts):
                if not video_flag:
                    if len(sample_path) == 1:
                        conversation = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": {"image_path": sample_path[0]}},
                                    {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\nQuery: " + query},
                                ]
                            },
                        ]
                    else:
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Image1:"},
                                    {"type": "image", "image": {"image_path": sample_path[0]}},
                                    {"type": "text", "text": "Image2:"},
                                    {"type": "image", "image": {"image_path": sample_path[1]}},
                                    {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\nQuery: " + query},
                                ]
                            }
                        ]
                else:
                    conversation = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": {"video_path": sample_path[0], "fps": 1, "max_frames": 4}},
                                {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\nQuery: " + query},
                            ]
                        },
                    ]

                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                output_ids = model.generate(**inputs, max_new_tokens=1024)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(f"Attempt {attempt + 1} response for key {key}:\n{response}")

                if isinstance(response, str) and response.strip():
                    try:
                        response_dict = json.loads(response)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON on attempt {attempt + 1} for key {key}: {response}")
                        continue

                    if all(k in response_dict for k in required_keys):
                        d[key] = {
                            "query": query,
                            "filename": sample,
                            **response_dict
                        }
                        valid_response = True
                        break  # Exit retry loop if successful

                time.sleep(5)

            if not valid_response:
                # Add default structure with empty fields
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
