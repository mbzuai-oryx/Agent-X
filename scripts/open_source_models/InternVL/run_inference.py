import math
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import json
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import time

tool_metadata_path = "/share/data/drive_1/hanan/multiagent_eval_data/toolmeta.json"
with open(tool_metadata_path, "r") as f:
    meta_data = json.load(f)
f.close()
meta_data = json.dumps(meta_data)

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


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map




IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

if __name__ == "__main__":

    required_keys = ["reasoning_steps", "final_answer"]
    reasoning_data_path ="/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/metadata/verified_data.json"
    with open(reasoning_data_path, "r") as g:
        reason_data = json.load(g)

    path = "/share/data/drive_1/hanan/InternVL/InternVL3-8B"

    model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


    final_results = []
    base_path = "/share/data/drive_1/hanan/multiagent_eval_data/VCA-Bench/samples"
    save_path = "internvl3_results_final.json"

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
            generation_config = dict(max_new_tokens=1024, do_sample=True)

            valid_response = False
            for attempt in range(5):
                try:
                    if not video_flag:
                        if len(sample_path) == 1:
                            pixel_values = load_image(sample_path[0], max_num=12).to(torch.bfloat16).cuda()
                            question = '<image>\n' + instruction_prompt + "\n\n" + "Query: " + query
                            response = model.chat(tokenizer, pixel_values, question, generation_config)
                        else:
                            pixel_values_1 = load_image(sample_path[0], max_num=12).to(torch.bfloat16).cuda()
                            pixel_values_2 = load_image(sample_path[1], max_num=12).to(torch.bfloat16).cuda()
                            pixel_values = torch.cat((pixel_values_1, pixel_values_2), dim=0)
                            question = '<image>\n' + instruction_prompt + "\n\n" + "Query: " + query
                            response = model.chat(tokenizer, pixel_values, question, generation_config)
                    else:
                        pixel_values, num_patches_list = load_video(sample_path[0], num_segments=8, max_num=1)
                        pixel_values = pixel_values.to(torch.bfloat16).cuda()
                        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                        question = video_prefix + instruction_prompt + "\n\n" + "Query: " + query
                        response, history = model.chat(
                            tokenizer, pixel_values, question, generation_config,
                            num_patches_list=num_patches_list, history=None, return_history=True
                        )

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
                            break  # exit retry loop
                except Exception as e_inner:
                    print(f"Error on attempt {attempt + 1} for key {key}: {e_inner}")
                    time.sleep(5)
                    continue

                time.sleep(5)

            if not valid_response:
                # Save fallback with empty fields
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
        except Exception as e:
            print(f"Exception for key {key}: {e}")
            continue


        
    with open('InternVL3_results_final_backup.json', 'w') as f:
        json.dump(final_results, f)
# if __name__ == "__main__":

    

#     path = "/share/data/drive_1/hanan/InternVL/InternVL2_5-8B"

#     model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True).eval().cuda()
#     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

#     ############################################ Image Inference ######################################################

#     # # set the max number of tiles in `max_num`
#     # pixel_values = load_image('/share/data/drive_1/hanan/video_gui_frames/RW_1/frame_00099.jpg', max_num=12).to(torch.bfloat16).cuda()
#     generation_config = dict(max_new_tokens=1024, do_sample=True)

#     # question = '<image>\nPlease describe the image shortly.'
#     # response = model.chat(tokenizer, pixel_values, question, generation_config)
#     # print(f'User: {question}\nAssistant: {response}')


#     ############################################ Video Inference ######################################################

#     video_path = '/share/data/drive_1/hanan/end.mp4'
#     pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
#     pixel_values = pixel_values.to(torch.bfloat16).cuda()


#     video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
#     question = video_prefix + 'What is happening in the video?'
#     # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
#     response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                 num_patches_list=num_patches_list, history=None, return_history=True)
#     print(f'User: {question}\nAssistant: {response}')


#     # question = 'Describe this video in detail.'
#     # response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#     #                             num_patches_list=num_patches_list, history=None, return_history=True)
#     # print(f'User: {question}\nAssistant: {response}')