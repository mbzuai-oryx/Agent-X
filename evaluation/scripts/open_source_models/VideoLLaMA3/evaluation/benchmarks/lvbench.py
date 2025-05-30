import json
import os
import re
from typing import Any, Dict, Union

from .base import BaseVideoEvalDataset


class LVBenchDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        video_folder = os.path.join(data_root, "video")
        json_file = os.path.join(data_root, "video_info.meta.jsonl")

        data_list = []
        with open(json_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    data_list.append(json.loads(line))

        for data in data_list:
            video_id = data["key"]
            for video_format in ["mp4", "avi", "mov", "mkv"]:
                video_path = os.path.join(video_folder, f"{video_id}.{video_format}")
                if os.path.exists(video_path):
                    break
            assert os.path.exists(video_path), f"Cannot find the video file: {video_path}"

            for qa_data in data["qa"]:
                data_dict[qa_data["uid"]] = {
                    # required fields for data loading
                    "video_path": video_path,
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": qa_data["question_type"],
                    "ground_truth": qa_data["answer"],
                    # custom fields for instruction generation and post processing
                    "question": qa_data["question"],
                }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        instruction = self.data_dict[data_id]["question"]
        instruction += "\nAnswer with the option\'s letter from the given choices directly."
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        matches = re.findall(r"\b[A-D]\b", response.upper())
        if len(matches) == 0:
            raise ValueError(f"Cannot find the answer in the response: {response}")
        return matches[0]
