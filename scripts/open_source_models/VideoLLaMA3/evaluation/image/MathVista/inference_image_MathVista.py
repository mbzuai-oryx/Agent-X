import argparse
import json
import math
import sys

sys.path.append('./')
import random

import numpy as np
import torch
from evaluation.register import INFERENCES
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from videollama3 import disable_torch_init


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MathVistaDataset(Dataset):
    def __init__(self, data_path, input_file, query_file, processor, num_chunks, chunk_idx):
        self.data = read_json(f'{data_path}/{input_file}')

        self.query_data = read_json(f'{data_path}/{query_file}')

        if num_chunks > 1:
            self.test_pids = get_chunk(list(self.data.keys()), num_chunks, chunk_idx)
        else:
            self.test_pids = list(self.data.keys())

        self.data_path = data_path
        self.processor = processor

    def __len__(self):
        return len(self.test_pids)

    def __getitem__(self, idx):
        pid = self.test_pids[idx]

        problem = self.data[pid]
        query = self.query_data[pid]
        image_path = f"{self.data_path}/{problem['image']}"

        image_tensor = self.processor(image_path)

        return {
            'pid':         pid,
            'problem':     problem,
            'image':       image_tensor,
            'query':       query
        }

def collate_fn(batch):
    pid = [x['pid'] for x in batch]
    problem = [x['problem'] for x in batch]
    image = [x['image'] for x in batch]
    query = [x['query'] for x in batch]
    return pid, problem, image, query


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)

    model, processor, tokenizer = model_init(args.model_path)

    assert args.batch_size == 1, "Batch size must be 1 for inference"

    dataset = MathVistaDataset(args.data_path, args.input_file, args.query_file, processor['image'], args.num_chunks, args.chunk_idx)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    if args.num_chunks > 1:
        output_file = args.output_file.replace('.json', f'_{args.num_chunks}_{args.chunk_idx}.json')
    else:
        output_file = args.output_file

    results = {}

    for idx, (pid, problem, image, query) in enumerate(tqdm(dataloader)):
        pid = pid[0]
        problem = problem[0] if args.instructions_message == '' else f'{problem[0]} {args.instructions_message}'
        image_tensor = image[0]
        query = query[0]

        set_random_seed(2024)

        output = mm_infer(
            image_tensor,
            query,
            model=model,
            tokenizer=tokenizer,
            modal='image',
            do_sample=False,
        )

        results[pid] = problem
        results[pid]['query'] = query
        results[pid]['response'] = output

    save_json(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--data-path', help='', required=True)
    parser.add_argument('--input-file', type=str, default='testmini.json')
    parser.add_argument('--query-file', type=str, default='query.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--instructions-message", type=str, default='')


    args = parser.parse_args()

    run_inference(args)