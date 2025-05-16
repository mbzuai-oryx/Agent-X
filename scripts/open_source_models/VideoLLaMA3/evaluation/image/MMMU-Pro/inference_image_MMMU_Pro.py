import argparse
import ast
import json
import math
import os
import re
import sys

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

sys.path.append('./')
import random

import numpy as np
from evaluation.register import INFERENCES
from videollama3 import disable_torch_init

MAX_RETRY = 5

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>\n"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    if args.interleaved:
        question = replace_images_tokens(question)
        parsed_options = replace_images_tokens(parsed_options)
    question = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    return question

def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    # return replace_images_tokens(question)
    return question

def origin_mmmu_doc_to_visual(doc):
    visual = []
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    prompt = f"{question}\n{parsed_options}\n{prompt_config['standard']}"
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens]))
    # for i in range(1,8):
    #     if not doc[f'image_{i}']:
    #         break
    #     visual.append(doc[f'image_{i}'])
    for image_token in image_tokens:
        visual.append(doc[image_token])
    return visual

def vision_mmmu_doc_to_visual(doc):
    return [doc['image']]

def process_prompt(data):
    if args.setting == 'standard':
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif args.setting == 'vision':
        prompt = prompt_config['vision']
        images = vision_mmmu_doc_to_visual(data)

    return (prompt, images)

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for output, data in results:
            data['response'] = output
            data = {k: v for k, v in data.items() if not k.startswith('image_')}
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(args):

    # Load processor and model
    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor, tokenizer = model_init(args.model_path)

    # Load prompt configuration
    global prompt_config
    with open("./evaluation/image/MMMU-Pro/prompts.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)[args.mode]

    dataset = load_dataset(args.data_path, args.setting, split='test')

    if args.num_chunks > 1:
        index_list = get_chunk(list(range(len(dataset))), args.num_chunks, args.chunk_idx)
        dataset = [dataset[idx] for idx in index_list]
    else:
        pass

    if args.num_chunks > 1:
        output_file = args.output_file.replace('.jsonl', f'_{args.num_chunks}_{args.chunk_idx}.jsonl')
    else:
        output_file = args.output_file.replace('.jsonl', f'_1_0.jsonl')

    print(f"Begin processing {args.setting}")

    results = []

    for idx, data in enumerate(tqdm(dataset, desc=f"Processing {args.setting}"), start=1):
        question, images = process_prompt(data)
        if not args.interleaved:
            num_image = len(images)
            image_tokens = ['<image>\n'] * num_image
            image_tokens = " ".join(image_tokens)
            question = f'{image_tokens}{question}'
        # else:
        #     # check how many <image> tokens in the question
        #     num_image = question.count('<image>')
        #     if num_image != len(images):
        #         # add one at the beginning
        #         question = f'<image>\n{question}'
        # add picture content
        image_list = []
        for _ in images:
            image_list.append(_.convert('RGB'))
        image_tensor = processor["image"](image_list)

        set_random_seed(2024)

        decoded_output = ""
        retry_count = 0
        max_retries = MAX_RETRY

        while not decoded_output and retry_count < max_retries:
            try:
                output = mm_infer(
                    image_tensor,
                    question,
                    model=model,
                    tokenizer=tokenizer,
                    modal='image',
                    do_sample=False,
                )
                decoded_output = output
                # print (decoded_output)
                if not decoded_output:
                    retry_count += 1
                    print(f"Retry {retry_count}/{max_retries} for {args.setting} due to empty output.")

            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{max_retries} for {args.setting} due to error: {str(e)}")

        if decoded_output:
            results.append(decoded_output)
        else:
            results.append('')
            print(f"Failed to get a non-empty output after {max_retries} retries for {args.setting}.")

    save_results_to_file(zip(results, dataset), output_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='/mnt/data/EVAL_BENCH/IMAGE/MMMU_Pro')
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--mode', help='', required=True)
    parser.add_argument('--setting', type=str, default='validation')
    parser.add_argument("--interleaved", action='store_true')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    global args

    args = parser.parse_args()

    main(args)
