import argparse
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from multiple_choice import match_multiple_choice
from PIL import Image
from tqdm import tqdm

sys.path.append('./')
import random

import numpy as np
from evaluation.register import INFERENCES
from videollama3 import disable_torch_init

disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def analyze_answer(d, gpt_answer, all_choices):
    """
    Extracts the multiple-choice answer from a long paragraph of model output.
    """
    try:
        intersect = list(set(all_choices).intersection(set(gpt_answer.split())))
        intersect_last = list(set(all_choices).intersection(set(gpt_answer.split('\n\n')[-1].split())))
        if gpt_answer in ["A", "B", "C", "D", "E"]:
            prediction = "(" + gpt_answer + ")"
        elif gpt_answer in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            prediction = gpt_answer
        elif (len(intersect) != 1 and len(intersect_last) != 1) or len(intersect) < 1:
            choices = ['(A)', '(B)', '(C)', '(D)', '(E)']
            options = '\n'.join([f'{choices[i]} {d["choices"][i]}' for i in range(len(d['choices']))])
            extracted_answer = match_multiple_choice(f"{d['question']}\nSelect from the following choices", options, gpt_answer)
            prediction = extracted_answer
        else:
            if len(intersect_last) == 1:
                intersect = intersect_last
                gpt_answer = gpt_answer.split('\n\n')[-1]
            prediction = intersect[0]
        return prediction
    except Exception as e:
        pass

def concat_images_horizontally_with_margin(image_filenames, output_filename, margin=10):
    """
    Concatenates images horizontally with a specified margin between images.
    """
    images = [Image.open(filename) for filename in image_filenames]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))

    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one
    new_image.save(output_filename)  # Save the result

def load_prompt(task_name, d, image_folder):
    """
    Loads the prompt and images, saves the images to a folder, and returns image paths and the prompt.
    """
    image_paths = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in d and d[k]:
            image = d[k]
            image_path = f'{image_folder}/{d["idx"]}_{k[-1]}.jpg'
            image.save(image_path)
            image_paths.append(image_path)
    prompt = d['prompt']
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    if 'blip' in model_path:
        prompt += '\nAnswer:'
    return image_paths, prompt

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def query_model(task_name, model_save_path):
    """
    Loads the dataset, queries the model, and saves the result to a JSON Lines file.
    """
    dataset_name = '/mnt/data/EVAL_BENCH/IMAGE/BLINK'
    # model_save_path = '_'.join(model_path.split('/')[-2:])

    # Modify output_path to use .jsonl extension
    if args.num_chunks > 1:
        output_path = f'{output_save_folder}/{model_save_path}/{task_name.replace("_", " ")}_{args.num_chunks}_{args.chunk_idx}.jsonl'
    else:
        output_path = f'{output_save_folder}/{model_save_path}/{task_name.replace("_", " ")}_1_0.jsonl'
    os.makedirs(f'{output_save_folder}/{model_save_path}', exist_ok=True)

    # Image folder can be shared since each process handles unique data points
    image_folder = f'{image_save_folder}/{task_name}_images'
    os.makedirs(image_folder, exist_ok=True)

    # Open the output file in append mode
    with open(output_path, 'a') as outfile:
        # Check if output file exists and is not empty, then load existing indices
        existing_indices = set()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        existing_indices.add(data['idx'])
                    except json.JSONDecodeError:
                        pass  # Skip invalid lines

        for split in ['val', 'test']:
            test_data = load_dataset(dataset_name, task_name)[split]
            # Split test_data
            if args.num_chunks > 1:
                index_list = get_chunk(list(range(len(test_data))), args.num_chunks, args.chunk_idx)
                test_data = [test_data[idx] for idx in index_list]
            for orig_d in tqdm(test_data):
                idx = orig_d['idx']
                if idx in existing_indices:
                    continue  # Skip already processed entries
                gold_answer = orig_d['answer']
                all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(orig_d['choices'])]
                image_paths, prompt = load_prompt(task_name, orig_d, image_folder)
                # Model inference
                set_random_seed(2024)
                image_list = []
                for _ in image_paths:
                    image_list.append(Image.open(_).convert('RGB'))
                image_tensor = processor["image"](image_list)
                num_image = len(image_paths)
                image_tokens = ['<image>\n'] * num_image
                image_tokens = " ".join(image_tokens)
                prompt = f'{image_tokens}{prompt}'
                gpt_answer = mm_infer(
                    image_tensor,
                    prompt,
                    model=model,
                    tokenizer=tokenizer,
                    modal='image',
                    do_sample=False,
                )
                prediction = analyze_answer(orig_d, gpt_answer, all_choices)
                # Write result as a JSON line
                result = {
                    'idx': idx,
                    'split': split,
                    'answer': gold_answer,
                    'full_prediction': gpt_answer,
                    'prediction': prediction
                }
                outfile.write(json.dumps(result) + '\n')

def eval_task(task_name, model_save_path):
    output_results = []
    # model_save_path = '_'.join(model_name.split('/')[-2:])
    # Modify output_path to include all chunked files
    if args.num_chunks > 1:
        output_files = [f'{output_save_folder}/{model_save_path}/{task_name.replace("_", " ")}_{args.num_chunks}_{i}.jsonl' for i in range(args.num_chunks)]
    else:
        output_files = [f'{output_save_folder}/{model_save_path}/{task_name.replace("_", " ")}_1_0.jsonl']

    for output_file in output_files:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        output_results.append(data)
                    except json.JSONDecodeError:
                        pass  # Skip invalid lines

    # Calculate accuracy
    accu = {'val': 0, 'test': 0}
    counts = {'val': 0, 'test': 0}
    for d in output_results:
        split = d['split']
        counts[split] = counts.get(split, 0) + 1
        if d['answer'] == d['prediction']:
            accu[split] = accu.get(split, 0) + 1

    print('-'*50)
    print(f'Task {task_name} Performance')
    for split in ['val', 'test']:
        if counts.get(split, 0) > 0:
            accuracy = round(accu[split] / counts[split] * 100, 2)
            print(f'{split} accuracy: {accuracy}%')
        else:
            print(f'No data for split {split}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/path/to/your/model', help="Select the model path")
    parser.add_argument("--model_name", type=str, help="Select the model name")
    parser.add_argument("--task_name", type=str, default='all', help="Select the task name")
    parser.add_argument("--output_path", type=str, default='/path/to/output', help="Select the output path")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total number of chunks to split the dataset")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Index of the current chunk")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    print(f'Using model: {model_path}')

    # Load processor and model
    disable_torch_init()
    model_init, mm_infer = INFERENCES(model_path)
    model, processor, tokenizer = model_init(model_path)

    image_save_folder = '/mnt/data/EVAL_BENCH/IMAGE/BLINK/BLINK_saved_images'
    output_save_folder = args.output_path
    dataset_name = '/mnt/data/EVAL_BENCH/IMAGE/BLINK'  # Use the Hugging Face dataset name

    need_disclaimer_tasks = ['Forensic_Detection', 'Jigsaw', 'Art_Style']
    if args.task_name == 'all':
        subtasks = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']
    else:
        subtasks = [args.task_name]

    for task_name in subtasks:
        query_model(task_name, args.model_name)
        eval_task(task_name, args.model_name)