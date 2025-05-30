
import argparse
import json
import math
import os
import re
import sys

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('./')
import random

import numpy as np
import torch
from evaluation.image.MMMU.data_utils import CAT_SHORT2LONG
from evaluation.image.MMMU.eval_utils import parse_multi_choice_response
from evaluation.register import INFERENCES
from videollama3 import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class MMMUDataset(Dataset):

    def __init__(self, data_path, vis_processors, split, config, num_chunks, chunk_idx, interleaved_format=True):
        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        self.dataset = concatenate_datasets(sub_dataset_list)
        
        self.interleaved_format = interleaved_format

        if num_chunks > 1:
            self.index_list = get_chunk(list(range(len(self.dataset))), num_chunks, chunk_idx)
        else:
            self.index_list = list(range(len(self.dataset)))

        self.vis_processors = vis_processors
        self.config = config

    def __len__(self):
        return len(self.index_list)

    def parse_img_path(self, text):
        matches = re.findall("<image (.*?)>", text)
        return matches

    def process_single_sample(self, data):
        question = data['question']
        
        o_imgs_paths = []

        for option in eval(data['options']):
            current_o_imgs_paths = self.parse_img_path(option)
            for img_path in current_o_imgs_paths:
                o_imgs_paths.append(f'image_{img_path}')
        
        pattern = ""
        for idx in range(1, 8):
            pattern += f'<image {idx}>|'
        
        pattern = rf"{pattern[:-1]}"
        
        # find the all matches
        matches = [(match.start(), match.group()) for match in re.finditer(pattern, question)]

        # get return image list
        return_image_list = []
        for start, match in matches:
            img_idx = match[-2]
            return_image_list.append(f'image_{img_idx}')
            
        if len(o_imgs_paths) > 0:
            return_image_list += o_imgs_paths
        
        if self.interleaved_format:
            for idx in range(1, 8):
                question = question.replace(f'<image {idx}>', '<image>\n')
            
            return_image_list = [data[image_path] for image_path in return_image_list]
            
            return_options = data['options']
            if len(o_imgs_paths) > 0:
                for idx in range(1, 8):
                    return_options = return_options.replace(f'<image {idx}>', '<image>')
            
            return {'id': data['id'], 'question': question, 'options': return_options, 'answer': data['answer'],
                    'image': return_image_list, 'question_type': data['question_type']}
        else: 
            return_image_list = sorted(list(set(return_image_list)))
            
            return_image_list = [data[image_path] for image_path in return_image_list]
            
            num_image = len(return_image_list)
            
            image_tokens = ['<image>\n'] * num_image 
            image_tokens = " ".join(image_tokens)
            
            question = f'{image_tokens}{question}'
            
            return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
                    'image': return_image_list, 'question_type': data['question_type']}

    def construct_prompt(self, sample, config):
        question = sample['question']
        options = eval(sample['options'])
        example = ""
        if sample['question_type'] == 'multiple-choice':
            start_chr = 'A'
            prediction_range = []
            index2ans = {}
            for option in options:
                if '<image>' in option and self.interleaved_format:
                    option = option.replace('<image>', '<image>\n')
                prediction_range.append(start_chr)
                example += f"({start_chr}) {option}\n"
                index2ans[start_chr] = option
                start_chr = chr(ord(start_chr) + 1)
            empty_prompt_sample_structure = config['multi_choice_example_format']
            empty_prompt = empty_prompt_sample_structure.format(question, example)
            res_dict = {}
            res_dict['index2ans'] = index2ans
            res_dict['correct_choice'] = sample['answer']
            res_dict['all_choices'] = prediction_range
            res_dict['empty_prompt'] = empty_prompt
            if config['task_instructions']:
                res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
            else:
                res_dict['final_input_prompt'] = empty_prompt

            res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
        else:
            empty_prompt_sample_structure = config['short_ans_example_format']
            empty_prompt = empty_prompt_sample_structure.format(question)
            res_dict = {}
            res_dict['empty_prompt'] = empty_prompt
            if config['task_instructions']:
                res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
            else:
                res_dict['final_input_prompt'] = empty_prompt
            res_dict['gt_content'] = sample['answer']

        res_dict.update(sample)
        return res_dict

    def __getitem__(self, idx):
        sample = self.dataset[self.index_list[idx]]
        
        sample = self.process_single_sample(sample)
        
        sample = self.construct_prompt(sample, self.config)

        image_list = [img.convert('RGB') for img in sample['image']]
        if sample['image']:
            sample['image'] = self.vis_processors(image_list)

        return sample


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):

    return batch


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)

    model, processor, tokenizer = model_init(args.model_path)

    # load config and process to one value
    config = load_yaml(args.config_path)
    for key, value in config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            config[key] = value[0]

    dataset = MMMUDataset(args.data_path, processor["image"], args.split, config, args.num_chunks, args.chunk_idx, interleaved_format=args.interleaved)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    if args.num_chunks > 1:
        output_file = args.output_file.replace('.json', f'_{args.num_chunks}_{args.chunk_idx}.json')
    else:
        output_file = args.output_file

    answer_file = os.path.join(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    out_samples = {}

    for idx, sample in enumerate(tqdm(dataloader)):
        image_tensor = sample[0]['image']
        question = sample[0]['final_input_prompt'] if args.instructions_message == '' else f'{sample[0]["final_input_prompt"]} {args.instructions_message}'
        question_type = sample[0]['question_type']
        question_id = sample[0]['id']
        
        set_random_seed(2024)

        output = mm_infer(
            image_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal='image',
            do_sample=False,
        )

        if question_type == 'multiple-choice':
            all_choices = sample[0]['all_choices']
            index2ans = sample[0]['index2ans']
            pred_ans = parse_multi_choice_response(output, all_choices, index2ans)
        else:  # open question
            pred_ans = output

        out_samples[question_id] = pred_ans
        
    json.dump(out_samples, ans_file, indent = 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--data-path', help='', required=True)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--config-path', type=str, default="./evaluation/image/MMMU/config_file.yaml")
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--instructions-message", type=str, default='')
    parser.add_argument("--interleaved", action='store_true')

    args = parser.parse_args()

    run_inference(args)
