import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union

from .base import BaseImageEvalDataset, filter_metadata


class AI2DDataset(BaseImageEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"
    
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        json_file = os.path.join(data_root, "ai2d_test_vlmevalkit.jsonl")
        data_list = [json.loads(item.strip()) for item in open(json_file).readlines()]

        for data in data_list:
            question_id = data["question_id"]
            image_path = os.path.join(data_root, data['image'])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
            
            answer = ord(data["answer"])-65
            assert answer >= 0 and answer < 4, f"Wrong ground truth for file: {image_path}. Ground Truth: {data['answer']}"
            
            data_dict[question_id] = {
                # required fields for data loading
                "image_path": image_path,
                # required fields for evaluation
                "ground_truth": answer,
                "task_type": "test",
                # custom fields for instruction generation and post processing
                "question": data["question"],
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        # options = self.data_dict[data_id]["options"]
        letters = ['A', 'B', 'C', 'D']

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        pred_answer = re.findall('[\(\ ]*[A-D][\)\ ]*', response)

        find_flag = False
        if len(pred_answer) == 0:
            # No answer option is considered incorrect
            # for idx, opt in enumerate(options):
            #     opt = opt.strip()
            #     opt = opt.strip('.')
            #     if opt.lower() in response.lower():
            #         pred_idx = idx
            #         find_flag = True
            #         break
            pred_idx = None
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        # assert find_flag, f"Cannot find the answer in the options: {response}"
        return pred_idx
