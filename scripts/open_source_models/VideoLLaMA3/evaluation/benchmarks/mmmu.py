import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
from collections import defaultdict
import pandas as pd
import re
from datasets import load_dataset
import numpy as np


from .base import BaseImageEvalDataset, filter_metadata

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}


class MMMUDataset(BaseImageEvalDataset):
    def __init__(self, *args, **kwargs):
        super(BaseImageEvalDataset, self).__init__(*args, **kwargs)
        
        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            aggregated_data[data_id] = {
                    "image_path": meta_data["image_path"],
                    "data_ids": [data_id],
                }
        aggregated_data_list = [x for _, x in aggregated_data.items()]
        self._aggregated_data_list = aggregated_data_list[kwargs["split_idx"]::kwargs["num_splits"]]
        
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        
        for subject in CAT_SHORT2LONG.values():
            data_list = load_dataset(os.path.join(data_root, subject))["validation"]

            for data in data_list:
                question_id = data["id"]
                if 'validation' not in question_id:
                    continue
                question = data["question"]
                
                # when <image> appears in options
                o_imgs_paths = []
                for option in eval(data["options"]):
                    current_o_imgs_paths = re.findall("<image (.*?)>", option)
                    for img_path in current_o_imgs_paths:
                        o_imgs_paths.append(f'image_{img_path}')
                
                # when <image> appears in questions
                pattern = ""
                for idx in range(1, 8):
                    pattern += f'<image {idx}>|'
                pattern = rf"{pattern[:-1]}"
                # find the all matches
                matches = [(match.start(), match.group()) for match in re.finditer(pattern, question)]

                # get return image list
                return_image_list = []
                for start, match in matches:
                    img_idx = match[-2]
                    return_image_list.append(f'image_{img_idx}')
                if len(o_imgs_paths) > 0:
                    return_image_list += o_imgs_paths
                images = [data[image_id].convert('RGB') for image_id in return_image_list]
                                
                data_dict[question_id] = {
                    # required fields for data loading
                    "image_path": images, #[data[f"image_{img_id}"].convert('RGB') for img_id in range(1,8) if data[f"image_{img_id}"] is not None],
                    "image_ids": return_image_list,
                    "question_id": question_id,
                    # required fields for evaluation
                    "task_type": subject,
                    "ground_truth": data["answer"],
                    # custom fields for instruction generation and post processing
                    "question": question,
                    "explanation": data["explanation"],
                    "question_type": data["question_type"],
                    "options": eval(data["options"])
                }

        return data_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        aggregated_data = self._aggregated_data_list[idx]
        
        try:
            images = self.processor.load_images(aggregated_data["image_path"])
            image_inputs = self.processor.process_image(images, return_tensors="pt")
        except:
            traceback.print_exc()
            print(f"Failed to load images: {aggregated_data}")
            exit()
            
        text_inputs = []
        for data_id in aggregated_data["data_ids"]:
            instruction = self.generate_instruction(data_id)
            content = []
            # for idx in range(len(image_inputs["grid_sizes"])):
            #     content += [{"type": "text", "data": f"<image {idx}>:"}, {"type": "image"}]
            conversation = [
                {
                    "role": "user",
                    "content": content + [{"type": "text", "data": instruction}],
                }
            ]
            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text_inputs.append(self.processor.process_text(prompt, image_inputs["grid_sizes"], return_tensors="pt"))

        data = {
            "data_ids": aggregated_data["data_ids"],
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
        }

        return data
    
    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        if meta_data["question_type"] == "multiple-choice":
            options = '\n'.join([f"({chr(65+n)}):{o}" for n, o in enumerate(meta_data["options"])])
            instruction = f"{question}\n\n{options}\n\nAnswer with the option's letter from the given choices directly."
        else:
            instruction = f"{question}\n\nAnswer the question using a single word or phrase."
        for idx in range(1, 8):
            instruction = instruction.replace(f'<image {idx}>', f'<image {idx}>:<image>')
        # print(instruction, self.data_dict[data_id]["images"])
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        meta_data = self.data_dict[data_id]
        options = meta_data["options"]
            
        pattern = r"boxed\{([^}]*)\}"
        boxed = re.findall(pattern, response)
        if len(boxed) > 0:
            return boxed[0]
        
        if meta_data["question_type"] == "multiple-choice":
            all_choices = [chr(65+num) for num in range(len(meta_data["options"]))]
            for char in [',', '.', '!', '?', ';', ':', "'"]:
                response = response.strip(char)
            response = " " + response + " " # add space to avoid partial match

            index_ans = True
            ans_with_brack = False
            candidates = []
                    
            for choice in all_choices:  # e.g., (A) (B) (C) (D)
                if f'({choice})' in response:
                    candidates.append(choice)
                    ans_with_brack = True

            if len(candidates) == 0:
                for choice in all_choices: # e.g., A B C D
                    if f' {choice} ' in response:
                        candidates.append(choice)
            

            # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
            if len(candidates) == 0 and len(response.split()) > 5:
                for index, ans in enumerate(meta_data["options"]):
                    if ans.lower() in response.lower():
                        candidates.append(chr(65+index))
                        index_ans = False # it's content ans.

            if len(candidates) == 0:  # still not get answer, randomly choose one.
                pred_index = random.choice(all_choices)
            elif len(candidates) > 1:
                start_indexes = []
                if index_ans:
                    if ans_with_brack:
                        for can in candidates:
                            index = response.rfind(f'({can})')
                            start_indexes.append(index) # -1 will be ignored anyway
                        # start_indexes = [generated_response.index(f'({can})') for can in candidates]
                    else:
                        for can in candidates:
                            index = response.rfind(f" {can} ")
                            start_indexes.append(index)
                else:
                    for can in candidates:
                        index = response.lower().rfind(options[ord(can)-65].lower())
                        start_indexes.append(index)
                # get the last one
                pred_index = candidates[np.argmax(start_indexes)]
            else: # if only one candidate, use it.
                pred_index = candidates[0]
            # print(pred_index)
            return pred_index
        else:
            return response
    
    def evaluate(self, results: List[Dict[str, Any]]):
        if self.TASK_TYPES is None:
            samples = defaultdict(list)
        else:
            samples = {task_type: [] for task_type in self.TASK_TYPES}
        infos = []
        
        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            task_type = meta_data["task_type"]
            question = meta_data["question"]
            ground_truth = meta_data["ground_truth"]
            response = data["prediction"]
            # if task_type != 'multi_choice':
            #     response
                
            correct = response==ground_truth
            samples[task_type].append(correct)
            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "matching": correct,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )
            
        task_types = samples.keys()
        metrics = {x: sum(samples[x]) / len(samples[x]) * 100 for x in task_types}

        overall_samples = sum(samples.values(), [])
        overall_acc = sum(overall_samples) / len(overall_samples) * 100
        metrics["Overall"] = overall_acc

        infos = [metrics] + infos
        return metrics, infos
