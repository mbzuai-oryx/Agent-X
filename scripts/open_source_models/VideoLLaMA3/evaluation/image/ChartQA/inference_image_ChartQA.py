# https://github.dev/QwenLM/Qwen-VL/blob/master/eval_mm/EVALUATION.md

import argparse
import json
import math
import random
import sys

sys.path.append('./')
import numpy as np
import torch
from evaluation.register import INFERENCES
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from videollama3 import disable_torch_init


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


class ChartQADataet(Dataset):
    def __init__(self, data_dir, test, processor, num_chunks, chunk_idx):
        full_test_set = open(test).readlines()

        if num_chunks > 1:
            self.test = get_chunk(full_test_set, num_chunks, chunk_idx)
        else:
            self.test = full_test_set

        self.data_dir = data_dir
        self.processor = processor

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)

        image_path = f'{self.data_dir}/{image}'
        image_tensor = self.processor(image_path)

        return {
            'image_path': image_path,
            'image': image_tensor,
            'question': question,
            'question_id': question_id,
            'annotation': annotation
        }


def collate_fn(batch):
    image_path = [x['image_path'] for x in batch]
    image = [x['image'] for x in batch]
    question = [x['question'] for x in batch]
    question_id = [x['question_id'] for x in batch]
    annotation = [x['annotation'] for x in batch]

    return image_path, image, question, question_id, annotation


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)

    model, processor, tokenizer = model_init(args.model_path)

    assert args.batch_size == 1, "Batch size must be 1 for inference"

    dataset = ChartQADataet(data_dir = args.data_dir,
                         test = f'{args.data_dir}/{args.test_file}',
                         processor = processor['image'],
                         num_chunks = args.num_chunks,
                         chunk_idx = args.chunk_idx)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    output_file = args.output_file.replace('.json', args.test_file).replace('jsonl', 'json')
    if args.num_chunks > 1:
        output_file = output_file.replace('.json', f'_{args.num_chunks}_{args.chunk_idx}.json')
    # else:
    #     output_file = args.output_file

    results = {}
    for idx, (image_path, image, question, question_id, annotation) in enumerate(tqdm(dataloader)):
        image_path = image_path[0]
        image_tensor = image[0]
        query = question[0] if args.instructions_message == '' else f'{question[0]}\n{args.instructions_message}'
        question_id = question_id[0]
        annotation = annotation[0]

        set_random_seed(2024)

        output = mm_infer(
            image_tensor,
            query,
            model=model,
            tokenizer=tokenizer,
            modal='image',
            do_sample=False,
        )

        temp_output = {}
        temp_output['image'] = image_path
        temp_output['question'] = query
        temp_output['answer'] = output
        temp_output['annotation'] = annotation
        results[question_id] = temp_output

    save_json(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--data-dir', help='', required=True)
    parser.add_argument('--test-file', type=str, default='test.jsonl')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--instructions-message", type=str, default="Answer the question using a single word or phrase.")

    args = parser.parse_args()

    run_inference(args)