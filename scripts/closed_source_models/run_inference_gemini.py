import os
import cv2
import tempfile
import shutil
import google.generativeai as genai
from PIL import Image

# Configure Gemini API
#genai.configure(api_key="") # Or use `GOOGLE_API_KEY` as env var
import cv2
import os
import time
import os
import json
import time
import base64
import ast 
import re
import argparse
parser = argparse.ArgumentParser()

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

def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return {
        "mime_type": "image/jpeg",
        "data": encoded
    }

def extract_video_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()
    return frame_paths, temp_dir

def gemini_infer(prompt, input_path, is_video=False):
    if is_video:
        frame_paths, temp_dir = extract_video_frames(input_path)
    else:
        frame_paths = [input_path]
        temp_dir = None

    # Encode images in base64
    base64_images = [encode_image_base64(p) for p in frame_paths]

    # Compose parts: list of dicts and final text
    parts = base64_images + [prompt]

    response = model.generate_content(parts)

    if temp_dir:
        shutil.rmtree(temp_dir)

    return response.text


## model_name = gemini-1.5-pro or gemini-2.5-pro-preview-05-06


if __name__=="__main__":

    final_results = []

    # Adding arguments
    parser.add_argument("--save_path",default="gemini_final_results.json", help = "path to Output file")
    parser.add_argument("--base_path", default="./AgentX/files",help = "path to data folder")
    parser.add_argument("--tool_data_path", default="./AgentX/tools_metadata.json", help = "path to tool metadata json file")
    parser.add_argument("--gt_data_path", default="./AgentX/data.json", help = "path to ground truth json file")
    parser.add_argument("--gemini_type", default="gemini-2.5-pro-preview-05-06",choices=["gemini-1.5-pro", "gemini-2.5-pro-preview-05-06"], help = "choice of gemini models")


    # Read arguments from command line
    args = parser.parse_args()

    model = genai.GenerativeModel(model_name = args.gemini_type, system_instruction = instruction_prompt)

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
             "Each reasoning step should include: \n" \
             + p

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
            
            if not video_flag:
                parts = [encode_image_base64(p) for p in sample_path]
                parts.append("Query: " + query)
                model_response = model.generate_content(parts)


            else:
                frame_paths, temp_dir = extract_video_frames(sample_path[0])
                base64_images = [encode_image_base64(p) for p in frame_paths]
                parts = base64_images + ["Query: " + query]
                model_response = model.generate_content(parts)



            response = model_response.text.strip()
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
            with open(args.save_path, "w") as f:
                json.dump(final_results, f, indent=2)
            time.sleep(5)
            if video_flag:
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Exception for key {key}: {e}")
            continue