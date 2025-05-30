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


class MMEDataset(Dataset):
    def __init__(self, question_file, image_folder, processor, num_chunks, chunk_idx):

        questions = [json.loads(q) for q in open(question_file, "r")]

        self.questions = get_chunk(questions, num_chunks, chunk_idx)

        self.processor = processor

        self.image_folder = image_folder

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = f'{self.image_folder}/{line["image"]}'
        question = line["text"]
        question_id = line["question_id"]

        image_tensor = self.processor(image_file)

        return {
            'image': image_tensor,
            'question': question,
            'question_id': question_id
        }


def collate_fn(batch):
    image = [x['image'] for x in batch]
    question = [x['question'] for x in batch]
    question_id = [x['question_id'] for x in batch]

    return image, question, question_id


def run_inference(args):
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)

    model, processor, tokenizer = model_init(args.model_path)

    assert args.batch_size == 1, "Batch size must be 1 for inference"

    dataset = MMEDataset(args.question_file, args.image_folder, processor['image'],  args.num_chunks, args.chunk_idx)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    if args.num_chunks > 1:
        output_file = args.output_file.replace('.jsonl', f'_{args.num_chunks}_{args.chunk_idx}.jsonl')
    else:
        output_file = args.output_file.replace('.jsonl', f'_1_0.jsonl')

    ans_file = open(output_file, "w")

    for idx, (image, question, question_id) in enumerate(tqdm(dataloader)):
        image_tensor = image[0]
        question = question[0]
        question_id = question_id[0]

        set_random_seed(2024)

        output = mm_infer(
            image_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal='image',
            do_sample=False,
        )

        ans_file.write(json.dumps({
            'question_id': question_id,
            "prompt": question,
            "text": output
        }) + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--question-file', help='', required=True)
    parser.add_argument('--image-folder', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, required=False, default=8)
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.')

    args = parser.parse_args()

    run_inference(args)