import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
from collections import defaultdict
from sklearn.metrics import accuracy_score

from .base import BaseImageEvalDataset, filter_metadata

eval_type_dict = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR",
    "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]

class MMEDataset(BaseImageEvalDataset):

    TASK_TYPES: List[str] = [task_type for task_type in eval_type_dict]
    
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        json_file = os.path.join(data_root, "mme.jsonl")
        data_list = [json.loads(q) for q in open(json_file, "r")]
        for task_type in eval_type_dict:
            image_dir = os.path.join(data_root, "MME_Benchmark_release_version", task_type)
            if os.path.exists(os.path.join(image_dir, "images")):
                gt_dir = os.path.join(image_dir, "questions_answers_YN")
                image_dir = os.path.join(image_dir, "images")
            else:
                gt_dir = image_dir
            for file in os.listdir(image_dir):
                if file.endswith('jpg') or file.endswith('png'):
                    image_path = os.path.join(image_dir, file)
                    for line in open(os.path.join(gt_dir, file.replace("jpg", "txt").replace("png", "txt"))):
                        question, answer = line.strip().split('\t')
                        question_id = task_type+file+answer
                        assert answer in ["Yes", "No"], f"Answer Wrong: {answer}"
                        data_dict[question_id] = {
                            # required fields for data loading
                            "image_path": image_path,
                            # required fields for evaluation
                            "task_type": task_type,
                            "ground_truth": answer,
                            # custom fields for instruction generation and post processing
                            "question": question,
                        }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"].replace(" Please answer yes or no.", "\nAnswer the question using a single word or phrase.")
        instruction = f'{question}\nPlease answer yes or no.'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        if 'yes' in response.lower():
            return "Yes"
        elif 'no' in response.lower():
            return "No"
        else:
            return "other"
    
    def evaluate(self, results: List[Dict[str, Any]]):
        if self.TASK_TYPES is None:
            samples = defaultdict(list)
        else:
            samples = {task_type: [[],[],0] for task_type in self.TASK_TYPES}
        infos = []
        
        label_map = {
            "Yes": 1,
            "No": 0,
            "other": -1,
        }
        

        for idx, data in enumerate(results):
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            ground_truth = meta_data["ground_truth"]
            task_type = meta_data["task_type"]
            matching = data["prediction"] == meta_data["ground_truth"]

            samples[task_type][0].append(label_map[data["prediction"]])
            samples[task_type][1].append(label_map[meta_data["ground_truth"]])
            if idx % 2 == 1:
                if infos[-1]["matching"] and matching:
                    samples[task_type][2] += 1
            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "matching": matching,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        task_types = samples.keys()
            
        metrics = {x: accuracy_score(samples[x][0], samples[x][1])*100 + samples[x][2]*200/len(samples[x][0]) for x in task_types}

        types = {
            "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
            "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
        }
        metrics["Perception"] = sum([metrics[x] for x in types["Perception"]])
        metrics["Cognition"] = sum([metrics[x] for x in types["Cognition"]])
        metrics["Overall"] = sum([metrics[x] for x in task_types])

        infos = [metrics] + infos
        return metrics, infos