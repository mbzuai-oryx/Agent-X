import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
from collections import defaultdict

from .base import BaseImageEvalDataset, filter_metadata


class GQADataset(BaseImageEvalDataset):

    # TASK_TYPES: List[str] = [task_type for task_type in TASKS]
    BENCHMARK_TYPE: str = "mcqa"
    
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        json_file = os.path.join(data_root, "llava_gqa_testdev_balanced.jsonl")
        data_list = [json.loads(item.strip()) for item in open(json_file).readlines()]
        with open(os.path.join(data_root, "data", "testdev_balanced_questions.json"), 'r') as f:
            answers = json.load(f)

        for data in data_list:
            image_path = os.path.join(data_root, "data", "images", data['image'])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
            question_id = data["question_id"]
                            
            data_dict[question_id] = {
                # required fields for data loading
                "image_path": image_path,
                # required fields for evaluation
                "task_type": data["category"],
                "ground_truth": answers[question_id]["answer"].lower(),
                # custom fields for instruction generation and post processing
                "question": data["text"],
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        return response.strip().strip(".").lower()
    
    # def evaluate(self, results: List[Dict[str, Any]]):
    #     if self.TASK_TYPES is None:
    #         samples = defaultdict(list)
    #     else:
    #         samples = {task_type: [] for task_type in self.TASK_TYPES}
    #     infos = []
        
    #     for data in results:
    #         data = deepcopy(data)
    #         meta_data = deepcopy(self.data_dict[data["data_id"]])
    #         task_type = meta_data["task_type"]
    #         dataset_name = meta_data["dataset_name"]
    #         ground_truth = meta_data["ground_truth"]
    #         response = data["prediction"]
            
    #         correct = False
    #         if dataset_name == "HME100k":
    #             if type(ground_truth)==list:
    #                 for j in range(len(ground_truth)):
    #                     answer = ground_truth[j].strip().replace("\n"," ").replace(" ","")
    #                     response = response.strip().replace("\n"," ").replace(" ","")
    #                     if answer in response:
    #                         correct = True
    #             else:
    #                 ground_truth = ground_truth.strip().replace("\n"," ").replace(" ","")
    #                 response = response.strip().replace("\n"," ").replace(" ","")
    #                 if ground_truth in response:
    #                         correct = True
    #         else:
    #             if type(ground_truth)==list:
    #                 for j in range(len(ground_truth)):
    #                     answer = ground_truth[j].lower().strip().replace("\n"," ")
    #                     response = response.lower().strip().replace("\n"," ")
    #                     if answer in response:
    #                         correct = True
    #             else:
    #                 ground_truth = ground_truth.lower().strip().replace("\n"," ")
    #                 response = response.lower().strip().replace("\n"," ")
    #                 if ground_truth in response:
    #                     correct = True
                
    #         samples[task_type].append(correct)
    #         infos.append(
    #             {
    #                 **data,
    #                 "ground_truth": ground_truth,
    #                 "matching": correct,
    #                 "task_type": task_type,
    #                 "meta_data": filter_metadata(meta_data),
    #             }
    #         )
            
    #     task_types = samples.keys()
    #     metrics = {x: sum(samples[x]) for x in task_types}

    #     overall_samples = sum(samples.values(), [])
    #     overall_acc = sum(overall_samples)
    #     metrics["Overall"] = overall_acc

    #     infos = [metrics] + infos
    #     return metrics, infos
    