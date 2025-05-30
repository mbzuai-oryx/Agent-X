import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
from collections import defaultdict
from datasets import load_dataset
from openai import AzureOpenAI

from .base import BaseImageEvalDataset, filter_metadata

endpoint = os.getenv("ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

class RealWorldQADataset(BaseImageEvalDataset):

    # TASK_TYPES: List[str] = [task_type for task_type in TASKS]
    BENCHMARK_TYPE: str = "mcqa"
    
    def __init__(self, *args, **kwargs):
        super(BaseImageEvalDataset, self).__init__(*args, **kwargs)
        self.client = AzureOpenAI(
            azure_endpoint = endpoint, 
            api_key=subscription_key,  
            api_version="2024-05-01-preview"
        )
        
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

        data_list = load_dataset(data_root, split='test')

        for question_id, data in enumerate(data_list):
            data_dict[question_id] = {
                # required fields for data loading
                "image_path": data["image"].convert('RGB'),
                # required fields for evaluation
                "task_type": "test",
                "ground_truth": data["answer"].strip().lower(),
                # custom fields for instruction generation and post processing
                "question": data["question"],
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        response = response.strip().lower()
        if response[-1] == '.':
            response = response[:-1]
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
            ground_truth = meta_data["ground_truth"]
            task_type = meta_data["task_type"]
            
            if data["prediction"].strip() == ground_truth:
                correct = True
            elif 'true' in self.match(ground_truth, data["prediction"].strip()).lower():
                correct = True
            else:
                correct = False

            samples[task_type].append(correct)
            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "score": correct,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        task_types = samples.keys()
        metrics = {x: sum(samples[x]) / len(samples[x]) * 100 for x in task_types}

        overall_acc = sum(metrics.values()) / len(metrics)
        metrics["Overall"] = overall_acc

        infos = [metrics] + infos
        return metrics, infos
    
    
    def build_prompt(self, gt, prediction):
        """
        Builds the prompt for the GPT-3.5 turbo model to match an answer with several options of a single-choice question.

        If the GPT-3.5 model is unable to find a match, it will output (Z).
        Also, if the original prediction does not clearly lean towards any of the options, it will output (Z).

        Parameters:
        - question: String, the question.
        - options: String, the options. E.g. ['(A)', '(B)']
        - prediction: String, the answer. E.g. '(B)'
        """
        tmpl = (
            "You are an AI assistant. Help me determine whether my answer is correct compared to the ground truth."
            "Note that answers that are not exactly the same but have the same meaning should still be considered correct."
            "Please output according to the specified template."
            "Output template: True means correct, False means incorrect. Only output True or False.\n"
            "Example 1: \n"
            "Ground Truth: yes, Answer: no, Your output: False\n"
            "Example 2: \n"
            "Ground Truth: three, Answer: 3, Your output: True\n"
            "Example 3: \n"
            "Ground Truth: 2, Answer: two, Your output: True\n"
            "Ground Truth: {}, Answer: {}, Your output: "
        )
        return tmpl.format(gt, prediction)


    def get_chat_response(self, promot, n=1, patience=10000000, sleep_time=0):
        messages = [
            {"role": "user", "content": promot},
        ]
        # print("I am here")
        while patience > 0:
            patience -= 1
            try:
                response = self.interaction(self.client, messages)
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction[0] != "" and prediction[0] != None:
                        return prediction

            except Exception as e:
                if "Rate limit" not in str(e):
                    print(e)

                if "Please reduce the length of the messages" in str(e):
                    print("!!Reduce promot size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(promot) * 0.9)
                    new_start = len(promot) - new_size
                    promot = promot[new_start:]
                    messages = [
                        {"role": "user", "content": promot},
                    ]
                    
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""


    def interaction(self, client, message_text):
        completion = client.chat.completions.create(
            model=deployment,
            messages = message_text,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        return completion


    def match(self, gt, prediction):
        prompt = self.build_prompt(gt, prediction)
        retry_limit = 10
        
        for retry in range(retry_limit):
            try:
                extraction = self.get_chat_response(prompt, patience=10)
                return extraction
            except Exception as e:
                time.sleep(1)
        return 'False'
