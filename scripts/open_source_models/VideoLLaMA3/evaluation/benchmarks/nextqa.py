import json
import os
import re
from typing import Dict, Any, Union

import pandas as pd

from .base import BaseVideoEvalDataset


class NextQADataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        json_file = os.path.join(data_root, "test.csv")
        video_folder = os.path.join(data_root, "NExTVideo")

        vid_to_path = json.load(open(os.path.join(data_root, 'map_vid_vidorID.json')))

        records = pd.read_csv(json_file)

        # NOTE: copied from NEXTQA dataset codebase
        # map_name = {'CW': 'Why', 'CH': 'How', 'TN': 'Bef&Aft', 'TC': 'When', 'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other', 'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D'}

        type_mapping = {
            'DC': 'Descriptive',
            'DL': 'Descriptive',
            'DO': 'Descriptive',
            'CH': 'Causal',
            'CW': 'Causal',
            'TC': 'Temporal',
            'TP': 'Temporal',
            'TN': 'Temporal',
        }

        types = []
        for record in records.itertuples():
            answer = record.answer
            options = [record.a0, record.a1, record.a2, record.a3, record.a4]
            options = [opt for opt in options if not pd.isna(opt)]

            option_letters = []
            for option_idx, option in enumerate(options): 
                option_letters.append(f"{chr(ord('A') + option_idx)}")
                if option == answer:
                    answer_idx = option_idx

            data_dict[idx] = {
                # required fields for data loading
                "video_path": os.path.join(video_folder, vid_to_path[str(record.video)] + '.mp4'),
                "start_time": None,
                "end_time": None,
                # required fields for evaluation
                "task_type": type_mapping[record.type],
                "ground_truth": record.answer,
                # custom fields for instruction generation and post processing
                "question": record.question,
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
