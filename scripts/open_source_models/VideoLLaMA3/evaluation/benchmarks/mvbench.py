import json
import os
import re
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset


TASKS = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}


class MVBenchDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"
    TASK_TYPES: List[str] = [task_type for task_type in TASKS]

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        for task_name, task_info in TASKS.items():
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
                    "start_time": data["start"] if task_info[3] else None,
                    "end_time": data["end"] if task_info[3] else None,
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
                if opt.lower() in response.lower():
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
