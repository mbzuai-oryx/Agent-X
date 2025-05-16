import argparse
import os
import os.path as osp
import random
import traceback
from typing import Any, Dict, List, Union

import datetime
import json
import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sys
sys.path.append(".")
from videollama3 import disable_torch_init, model_init, mm_infer
from evaluation.benchmarks import build_dataset
from evaluation.register import INFERENCES
from evaluation.utils import CUDADataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)

    parser.add_argument("--data-root", "--data_root", type=str, required=True)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=8)

    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", "--max_frames", type=int, default=180)
    parser.add_argument("--max-visual-tokens", "--max_visual_tokens", type=int, default=None)

    parser.add_argument("--save-path", "--save_path", type=str, default=None)

    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_metrics(metrics: Dict[str, Any], benchmark: str):
    if all(isinstance(metric, dict) for metric in metrics.values()):
        for task_name, metric in metrics.items():
            show_metrics(metric, f"{benchmark}_{task_name}")
    elif all(isinstance(metric, (int, float)) for metric in metrics.values()):
        table = PrettyTable(["Task Type", "Accuracy"])
        for task_name, metric in metrics.items():
            table.add_row([task_name, round(metric, 2)])
        table.align["Task Type"] = "l"
        print(f"Results on {benchmark}:")
        print(table)
        print("\n")
    else:
        raise ValueError


def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()

    seed_everything()
    args = parse_args()

    disable_torch_init()
    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor = model_init(
        args.model_path,
        args.max_visual_tokens,
        device_map={"": f"cuda:{local_rank}"}
    )

    dataset = build_dataset(
        args.benchmark,
        data_root=args.data_root,
        processor=processor,
        num_splits=dist.get_world_size(),
        split_idx=global_rank,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    dataloader = CUDADataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=lambda x: x[0],  # asume the batch_size is always 1
        pin_memory=True,
    )
    modal = dataset.MODAL

    results = []
    for idx, data in enumerate(tqdm(dataloader, desc=f"Rank {global_rank}", total=len(dataloader), position=local_rank)):
        data_ids = data["data_ids"]
        text_inputs = data["text_inputs"]
        for data_id, text_input in zip(data_ids, text_inputs):
            try:
                data_dict = {**data["image_inputs"], **text_input}
                response = mm_infer(
                    data_dict,
                    model=model,
                    tokenizer=processor.tokenizer,
                    modal=modal,
                    do_sample=False,
                )
                prediction = dataset.process_response(data_id, response)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in data_id: {data_id}")
                exit(0)

            results.append(
                {
                    "data_id": data_id,
                    "response": response,
                    "prediction": prediction,
                }
            )
    
    assert len(results) == dataset.n_samples

    del model, data
    torch.cuda.empty_cache()

    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.gather_object(
        obj=results,
        object_gather_list=gathered_results if global_rank == 0 else None,
        dst=0,
    )

    if global_rank == 0:
        results = sum(gathered_results, [])
        metrics, infos = dataset.evaluate(results)

        print("\n" * dist.get_world_size())  # prevent unexpected progress bars
        show_metrics(metrics, args.benchmark)

        if args.save_path:
            os.makedirs(osp.dirname(args.save_path), exist_ok=True)
            if args.save_path.endswith(".json"):
                with open(args.save_path, "w") as f:
                    json.dump(infos, f, indent=4)
            elif args.save_path.endswith(".jsonl"):
                with open(args.save_path, "w") as f:
                    for info in infos:
                        f.write(json.dumps(info) + "\n")
            else:
                raise ValueError("Unsupported file format.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
