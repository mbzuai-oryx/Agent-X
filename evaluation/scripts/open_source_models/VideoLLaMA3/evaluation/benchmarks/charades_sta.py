import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset, filter_metadata


class CharadesSTADataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "temporal_grounding"

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        video_folder = os.path.join(data_root, "Charades_v1")
        json_file = os.path.join(data_root, "charades_annotations_test-random_prompt.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)

        for data in data_list:
            data_dict[data["question_id"]] = {
                # required fields for data loading
                "video_path": os.path.join(video_folder, data["video"]),
                "start_time": None,
                "end_time": None,
                # custom fields for instruction generation and post processing
                "question": data["conversation"][0]["content"],
                "ground_truth": data["timestamps"][0],
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        question = self.data_dict[data_id]["question"]
        instruction = question + "Please output the start and end timestamps in seconds."
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        # match "from x.x to y.y" or "x.x - y.y" 
        pattern = re.compile(r'(\d+\.?\d*|\d*\.\d+)\s*(?:-|to)\s*(\d+\.?\d*|\d*\.\d+)')
        matches = pattern.findall(response)
        if len(matches) > 0:
            intervals = [[float(start), float(end)] for start, end in matches]
        else:
            pattern = r'\d*\.?\d+'
            intervals = re.findall(pattern, response)
            if len(intervals) % 2 != 0:
                intervals.pop(0)
            intervals = [[float(intervals[i * 2]), float(intervals[i * 2 + 1])] for i in range(len(intervals) // 2)] 
        return intervals
