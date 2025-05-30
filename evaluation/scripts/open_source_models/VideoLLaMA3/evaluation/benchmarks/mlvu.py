import json
import os
import re
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset


MCQA_TASKS = {
    "TR:  (Topic Reasoning)":     ("7_topic_reasoning.json", "7_topic_reasoning"),
    "AR:  (Anomaly Recognition)": ("6_anomaly_reco.json", "6_anomaly_reco"),
    "NQA: (Needle QA)":           ("2_needle.json", "2_needle"),
    "ER:  (Ego Reasoning)":       ("3_ego.json", "3_ego"),
    "PQA: (Plot QA)":             ("1_plotQA.json", "1_plotQA"),
    "AC:  (Action Count)":        ("4_count.json",  "4_count"),
    "AO:  (Action Order)":        ("5_order.json",  "5_order"),
}

OQA_TASKS = {
    "VS:  (Video Summary)":        ("9_summary.json", "9_summary"),
    "SSC: (Sub-Scene Captioning)": ("8_sub_scene.json", "8_sub_scene"),
}


class MLVUDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"
    TASK_TYPES: List[str] = [task_type for task_type in MCQA_TASKS]

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        for task_name, task_info in MCQA_TASKS.items():
            json_file = os.path.join(data_root, "json", task_info[0])
            video_folder = os.path.join(data_root, "video", task_info[1])

            with open(json_file, 'r') as f:
                task_data_list = json.load(f)

            for data in task_data_list:
                answer = data["answer"]
                options = data["candidates"]

                option_letters = []
                for option_idx, option in enumerate(options): 
                    option_letters.append(f"{chr(ord('A') + option_idx)}")
                    if option == answer:
                        answer_idx = option_idx

                data_dict[idx] = {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, data["video"]),
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": task_name,
                    "ground_truth": answer_idx,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                    "options": options,
                    "option_letters": option_letters,
                }
                idx += 1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        option_letters = meta_data["option_letters"]
        options = meta_data["options"]

        option_string = ""
        for option_idx, (letter, option) in enumerate(zip(option_letters, options)):
            option_string += f"({letter}) {option}\n"
        instruction = f"Question: {question}\nOptions:\n{option_string}Answer with the option\'s letter from the given choices directly and only give the best option."

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        meta_data = self.data_dict[data_id]
        options = meta_data["options"]
        option_letters = meta_data["option_letters"]

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        pred_answer = re.findall(f'[\(,\ ]*[{option_letters[0]}-{option_letters[-1]}][\),\ ]*', response)

        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = option_letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        return pred_idx
