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


class InfoVQADataset(BaseImageEvalDataset):
    def load_data(self, data_root: str) -> Dict[int, Any]:
        self.set = "val"
        if self.set == "val":
            json_file = "infographicsVQA_val_v1.0_withQT.json"
        else:
            json_file = "infographicsVQA_test_v1.0.json"
        data_dict = {}

        json_file = os.path.join(data_root, json_file)
        with open(json_file, 'r') as f:
            data_list = json.load(f)['data']

        for data in data_list:
            question_id = data["questionId"]
            image_path = os.path.join(data_root, "images", data['image_local_name'])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
                            
            data_dict[question_id] = {
                # required fields for data loading
                "image_path": image_path,
                # required fields for evaluation
                "ground_truth": data["answers"] if self.set=="val" else None,
                "task_type": "val",
                # custom fields for instruction generation and post processing
                "question": data["question"],
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}\nAnswer the question with a single word or phrase.'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> str:
        return response
    
    def evaluate(self, results: List[Dict[str, Any]]):
        if self.set == "test":
            result_json = []
            for data in results:
                result_json.append({"questionId": data["data_id"], "answer": data["prediction"]})
            return {}, result_json
        
        if self.TASK_TYPES is None:
            samples = defaultdict(list)
        else:
            samples = {task_type: [] for task_type in self.TASK_TYPES}
        infos = []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            ground_truth = meta_data["ground_truth"]
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            task_type = meta_data["task_type"]
        
            values = []
            for answer in ground_truth:
                # preprocess both the answers - gt and prediction
                gt_answer = ' '.join(answer.strip().lower().split())
                det_answer = ' '.join(data["prediction"].strip().lower().split())

                dist = self.levenshtein_distance(gt_answer,det_answer)
                length = max( len(answer.upper()), len(data["prediction"].upper()) )
                values.append( 0.0 if length == 0 else float(dist) / float(length) )

            question_result = 1 - min(values)
            
            samples[task_type].append(question_result)
            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "score": question_result,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )
            
        task_types = samples.keys()
        metrics = {x: sum(samples[x]) / len(samples[x]) * 100 for x in task_types}

        infos = [metrics] + infos
        return metrics, infos
    
    
    def levenshtein_distance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
    