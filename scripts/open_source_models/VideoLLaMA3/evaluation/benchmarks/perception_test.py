import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset, filter_metadata


class PerceptionTestDataset(BaseVideoEvalDataset):

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        video_folder = os.path.join(data_root, "videos")
        json_file = os.path.join(data_root, "mc_question_test.json")
        with open(json_file, "r") as f:
            data_list = json.load(f).values()

        for data in data_list:
            video_id = data["metadata"]["video_id"]
            for video_format in ["mp4", "avi", "mov", "mkv"]:
                temp_path = os.path.join(video_folder, f"{video_id}.{video_format}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            assert os.path.exists(video_path), f"Cannot find the video file: {video_id}"

            for question in data["mc_question"]:
                question_id = question["id"]
                data_dict[f"{video_id}_{question_id}"] = {
                    # required fields for data loading
                    "video_path": video_path,
                    "start_time": None,
                    "end_time": None,
                    # custom fields for instruction generation and post processing
                    "question": question["question"],
                    "options": question["options"],
                    "video_id": video_id,
                    "question_id": question["id"],
                }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]
        instruction = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        letters = ['(A)', '(B)', '(C)']
        options = self.data_dict[data_id]["options"]

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        pred_answer = re.findall('\(*[A-C]\)*', response)

        if len(pred_answer) >= 1:
            pred_answer = pred_answer[0].strip()
            # if not pred_answer.startswith('('):
            pred_answer = pred_answer.strip('()')
            pred_answer = f'({pred_answer})'
            pred_idx = letters.index(pred_answer)
        else:
            tmp_options = [x.lower() for x in _options]
            assert response.lower() in tmp_options, f"Cannot find the answer in the options: {response}"
            tmp_options = [x.lower() for x in _options]
            pred_idx = tmp_options.index(response.lower())

        return pred_idx

    def evaluate(self, results: List[Dict[str, Any]]) -> (None, Dict[str, List[Dict[str, Any]]]):
        characters = string.ascii_letters + string.digits
        file_name = "perceptiontest-eval-" + "".join(random.choice(characters) for _ in range(16)) + ".json"
        save_path = os.path.join(".cache", file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        submission = {result["data_id"]: result["prediction"] for result in results}
        submission = defaultdict(list)
        for data in results:
            meta_data = self.data_dict[data["data_id"]]
            answer_id = data["prediction"]
            video_id = meta_data["video_id"]
            question_id = meta_data["question_id"]
            options = meta_data["options"]
            submission[video_id].append(
                {"id": question_id, "answer_id": answer_id, "answer": options[answer_id]}
            )

        # with open(save_path, "w") as f:
        #     json.dump(submission, f, indent=4)
        # os.system(f"evalai challenge 2091 phase 4156 submit --file {save_path} --private")

        # infos = []
        # for data in results:
        #     data = deepcopy(data)
        #     meta_data = deepcopy(self.data_dict[data["data_id"]])
        #     infos.append(
        #         {
        #             **data,
        #             "meta_data": filter_metadata(meta_data),
        #         }
        #     )
        # infos = [metrics] + infos

        # return {}, infos

        return {}, submission
