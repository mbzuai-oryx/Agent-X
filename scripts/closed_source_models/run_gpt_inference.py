import openai
import cv2
import os
import time
import os
import json
import time
import base64
import ast 
import re

tool_metadata_path = "/share/data/drive_1/hanan/multiagent_eval_data/toolmeta.json"
with open(tool_metadata_path, "r") as f:
    meta_data = json.load(f)
f.close()
meta_data = json.dumps(meta_data)

openai.api_key = "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"
client = openai.OpenAI(api_key="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")

p = """
    "reasoning_step_format": [
        {"task": "Describe the sub-task being performed."},
        {"tool_used": "Specify the tool selected and justify its choice."},
        {"tool_output": "Provide the tool's output."},
        {"thought": "Explain the significance of the output and how it contributes to answering the query."}
    ]

    "constraints": [
        "Use tools only when necessary and justify their usage.",
        "Ensure each step is self-contained and clearly explained.",
        "Maintain transparency in decision-making and reasoning."
    ]

    "final_output": {
        "final_answer": "Provide a clear and concise answer based on all previous steps", "justification": "Provide a justification for the final answer."    
    }

    "The output should be a dictionary as shown in the following example":\n

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
instruction_prompt = "You are an intelligent multi-modal agent. You are provided with:\n" \
             "- A text query\n" \
             "- An image or video\n" \
             "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"\
             f"{meta_data}\n\n" \
             "Your objective is to answer the query based on the given visual content " \
             "by choosing and using the most appropriate tools. You must reason step-by-step. " \
             "Each reasoning step should include: \n" \
             + p
            #   + json.dumps(p, indent=2)

reasoning_data_path ="/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/metadata/verified_data.json"
with open(reasoning_data_path, "r") as g:
    reason_data = json.load(g)

def extract_video_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to base64
        _, img_encoded = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(img_encoded).decode("utf-8")
        frames.append(f"data:image/jpeg;base64,{base64_frame}")
        frame_count += 1
    cap.release()
    return frames


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

final_results = []
#base_path = "/share/data/drive_1/hanan/multiagent_eval_data/10_samples"
base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"

save_path = "o4_mini_results_final.json"

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
        print(sample_path)
        if not video_flag:
            if len(sample_path) == 1:
                # file = client.files.create(
                #             file=open(sample_path[0], "rb"),
                #             purpose="vision"
                #         )
                # with open(sample_path[0], "rb") as f:
                #     file = openai.files.create(file=f, purpose="vision")
                base64_image = encode_image(sample_path[0])
                data_url = f"data:image/jpeg;base64,{base64_image}"
                # Run inference
                gpt_response = openai.chat.completions.create(
                    model="o4-mini",
                    messages=[{"role": "system", "content": instruction_prompt + '\n\n'},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": "Query: " + query}
                        ]}
                    ]
                )
                response = gpt_response.choices[0].message.content
                

            else:
                image_blocks = [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                        for image_path in sample_path
                    ]
                
                gpt_response = openai.chat.completions.create(
                        model="o4-mini",
                        messages=[{"role": "system", "content": instruction_prompt + '\n\n'},
                            {  
                                "role": "user",
                                "content": [{"type": "text", "text": "Query: " + query}] + image_blocks
                            }
                        ]
                    )
                response = gpt_response.choices[0].message.content


        else:

            video_path = sample_path[0]
            frames = extract_video_frames(video_path, num_frames=8)  # Extract first 5 frames

            # Run GPT-4o inference with video frames (multiple frames)
            gpt_response = openai.chat.completions.create(
                model="o4-mini",
                messages=[{"role": "system", "content": instruction_prompt + '\n\n'},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Query: " + query}
                        ] + [{"type": "image_url", "image_url": {"url": frame}} for frame in frames]
                    }
                ]
            )
            response = gpt_response.choices[0].message.content
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
        if isinstance(response, str):
            try:
                response_dict = json.loads(response)
                # if all(k in response_dict for k in required_keys):
                d[key] = {
                        "query": query,
                        "filename": sample,
                        **response_dict
                    }

            except json.JSONDecodeError:
                print(f"Invalid JSON on attempt")
                try:
                    response_dict = ast.literal_eval(response)
                    d[key] = {
                        "query": query,
                        "filename": sample,
                        **response_dict
                                    }
                except:

                    d[key] = {
                        "query": query,
                        "filename": sample,
                        "reasoning_steps": "",
                        "final_answer": {
                            "value": "",
                            "justification": ""
                        }
                    }
        else:
            d[key] = {
                    "query": query,
                    "filename": sample,
                    "reasoning_steps": "",
                    "final_answer": {
                        "value": "",
                        "justification": ""
                    }
                }
        print(response)
        final_results.append(d)
        with open(save_path, "w") as f:
            json.dump(final_results, f, indent=2)
        time.sleep(5)
    except Exception as e:
        print(f"Exception for key {key}: {e}")
        continue


    
with open('o4_mini_results_final_backup.json', 'w') as f:
    json.dump(final_results, f)


# === Example Usage ===

