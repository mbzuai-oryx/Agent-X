import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import json
import time

tool_metadata_path = "/share/data/drive_1/hanan/multiagent_eval_data/toolmeta.json"
with open(tool_metadata_path, "r") as f:
    meta_data = json.load(f)
f.close()
meta_data = json.dumps(meta_data)

# instruction_prompt = (
#         "You are a multi-modal intelligent agent. You are provided with:"
#         "- A text query,"
#         "- A single or multiple images or videos,"
#         "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"

#         f"{meta_data}\n\n"

#         "Your objective is to answer the query based on the given visual content "
#         "by choosing and using the most appropriate tools. You must reason step-by-step. "
#         "Each reasoning step should include: \n\n"

#     "reasoning_step_format:"
#         "{task: Describe the sub-task being performed.,"
#         "tool_used: Specify the tool selected and justify its choice.,"
#         "tool_output: Provide the tool's output.,"
#         "thought: Explain the significance of the output and how it contributes to answering the query.}\n\n"

#     "constraints:"
#         "Use only necessary tools and justify their usage.,"
#         "Ensure each step is self-contained and clearly explained.,"
#         "Maintain transparency in decision-making and reasoning.\n\n"

#     "final_output\n"
#         "final_answer: Provide a clear and concise answer to the query based on all previous steps.\n\n"

#     "The output should be a python dictionary as shown in the following example:\n"
#         "{"
#         "query:" "What is the man doing in the video?,"
#         "reasoning_steps:"
#         "["
#          "{"
#          "step:" "1,"
#             "task:" "Identify key frames containing activity.,"
#             "tool_used:" "Scene Segmentation Tool,"
#             "tool_output:" "Frames 40-80 show continuous motion.,"
#             "thought:"  "These frames likely contain the main action.},"
#             "{"
#             "step:" "2,"
#                 "task:" "Recognize the action happening in the selected frames.,"
#                 "tool_used:" "Action Recognition Model,"
#                 "tool_output:" "Man is playing guitar,"
#                 "thought:" "The model confirms that the man is engaged in a musical activity."
#                 "}"
#                 "],"
#         "final_answer:" "The man is playing a guitar in the video."
#         "}"
# )

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

instruction_prompt = "You are an intelligent multi-modal agent. You are provided with:\n" \
             "- A text query\n" \
             "- An image or video\n" \
             "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"\
             f"{meta_data}\n\n" \
             "Your objective is to answer the query based on the given visual content " \
             "by choosing and using the most appropriate tools. You must reason step-by-step. " \
             "Each reasoning step should include: \n\n" \
              + json.dumps(p, indent=2)

reasoning_data_path ="/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/metadata/verified_data.json"
with open(reasoning_data_path, "r") as g:
    reason_data = json.load(g)

# print(type(reason_data))

device = "cuda:0"
model_path = "/share/data/drive_1/hanan/VideoLLaMA3/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "video": {"video_path": "/share/data/drive_1/hanan/end.mp4", "fps": 1, "max_frames": 180}},
#             {"type": "text", "text": "What is happening in the video?"},
#         ]
#     },
# ]

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": {"image_path": "/share/data/drive_1/hanan/video_gui_frames/AE_4/frame_00000.jpg"}},
#             {"type": "text", "text": "Describe the image?"},
#         ]
#     },
# ]

# conversation = [
#     {"role": "system", "content": instruction_prompt[0]},
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": {"image_path": "/share/data/drive_1/hanan/multiagent_eval_data/10_samples/001_COCO_15.jpg"}},
#             {"type": "text", "text": instruction_prompt[0] + "\n\n" + "Query: " + "At what time and what news is being broadcasted on the TV and what is the object the TV is placed inside?"},
#         ]
#     },
# ]

final_results = []
#base_path = "/share/data/drive_1/hanan/multiagent_eval_data/10_samples"
base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"


required_keys = ["reasoning_steps", "final_answer"]
save = 0

for key, value in reason_data.items():
    d = {}
    try:
        video_flag = False
        data = reason_data[key][0]
        sample = data["file_path"]
        sample_list = [img.strip() for img in sample.split(",")]
        if sample_list[0].split(".")[1] in ["mp4", "avi", "mov"]:
            video_flag = True
        sample_path = [os.path.join(base_path, s) for s in sample_list]
        query = data["query"]

        valid_response = False
        for attempt in range(5):
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
                            {"type": "video", "video": {"video_path": sample_path[0], "fps": 1, "max_frames": 180}},
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

    except Exception as e:
        print(f"Exception for key {key}: {e}")
        continue

    # save += 1
    # if save == 10:
    #     break


# save = 0
# for key, value in reason_data.items():
#     d = {}
#     try:
#         video_flag = False
#         data = reason_data[key][0]
#         sample = data["file_path"]
#         sample_list = [img.strip() for img in sample.split(",")]
#         if sample_list[0].split(".")[1] in ["mp4", "avi", "mov"]:
#             video_flag = True
#         sample_path = [os.path.join(base_path,s) for s in sample_list]
#         query = data["query"]


#         if not video_flag:
#             if len(sample_path)==1:
#                 conversation = [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "image", "image": {"image_path": sample_path[0]}},
#                             {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\n" + "Query: " + query},
#                         ]
#                     },
#                 ]
#             else:
#                 conversation = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": "Image1: "},
#                         {"type": "image", "image": {"image_path": sample_path[0]}},
#                         {"type": "text", "text": "Image2: "},
#                         {"type": "image", "image": {"image_path": sample_path[1]}},
#                         {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\n" + "Query: " + query},
#                     ]
#                 }
#             ]


#         else:
#             # if len(sample_path)==1:
#             conversation = [
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "video", "video": {"video_path": sample_path[0], "fps": 1, "max_frames": 180}},
#                             {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\n" + "Query: " + query},
#                         ]
#                     },
#                 ]


#         inputs = processor(
#             conversation=conversation,
#             add_system_prompt=True,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         )
#         inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#         if "pixel_values" in inputs:
#             inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
#         output_ids = model.generate(**inputs, max_new_tokens=1024)
#         response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         print(response)
#         if isinstance(response, str) and response.strip():
#             try:
#                 response_dict = json.loads(response)
#             except json.JSONDecodeError:
#                 print(f"Invalid JSON for key {key}: {response}")
#                 response_dict = response  # or handle appropriately

#             # response_dict = json.loads(response)
#             d[key] = {
#             "query": query,
#             "filename": sample,
#             **response_dict  # Unpack the contents of response_dict directly
#         }
#         else:
#             d[key] = {
#             "query": query,
#             "filename": sample,
#             "reasoning_steps": "",
#             "final_answer": ""
#         }
#             response_dict = response
#         final_results.append(d)
#         time.sleep(5)
#     except Exception as e:
#         print(e)
#         # d  = {key: {"query": query, "filename":sample, "model_response": ""}}
#         # final_results.append(d)
#         continue
#     save+=1
#     if save==10:
#         break






# final_results = []
# #base_path = "/share/data/drive_1/hanan/multiagent_eval_data/10_samples"
# base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"
# for data in reason_data:
#     video_flag = False
#     sample = data["file_path"]
#     sample_list = [img.strip() for img in sample.split(",")]
#     if sample_list[0].split(".")[1] in ["mp4", "avi", "mov"]:
#         video_flag = True
#     sample_path = [os.path.join(base_path,s) for s in sample_list]
#     query = data["query"]


#     if not video_flag:
#         if len(sample_path)==1:
#             conversation = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": {"image_path": sample_path[0]}},
#                         {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\n" + "Query: " + query},
#                     ]
#                 },
#             ]
#         else:
#             conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Image1: "},
#                     {"type": "image", "image": {"image_path": sample_path[0]}},
#                     {"type": "text", "text": "Image2: "},
#                     {"type": "image", "image": {"image_path": sample_path[1]}},
#                     {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\n" + "Query: " + query},
#                 ]
#             }
#         ]


#     else:
#         # if len(sample_path)==1:
#         conversation = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "video", "video": {"video_path": sample_path[0], "fps": 1, "max_frames": 180}},
#                         {"type": "text", "text": json.dumps(instruction_prompt, indent=2) + "\n\n" + "Query: " + query},
#                     ]
#                 },
#             ]


#     inputs = processor(
#         conversation=conversation,
#         add_system_prompt=True,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     )
#     inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#     if "pixel_values" in inputs:
#         inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
#     output_ids = model.generate(**inputs, max_new_tokens=1024)
#     response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#     print(response)
#     d  = {"query": query, "filename":sample, "model_response": response}
#     final_results.append(d)
#     time.sleep(10)



with open('videollama3_results_final.json', 'w') as f:
    json.dump(final_results, f)
