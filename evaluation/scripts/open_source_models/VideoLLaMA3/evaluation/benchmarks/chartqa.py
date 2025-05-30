import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union, Optional
from collections import defaultdict
from openai import AzureOpenAI

from .base import BaseImageEvalDataset, filter_metadata

TASKS = {"augmented": "test_augmented_renamed.jsonl", "human": "test_human_renamed.jsonl"}

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


class ChartQADataset(BaseImageEvalDataset):

    TASK_TYPES: List[str] = [task_type for task_type in TASKS]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AzureOpenAI(
            azure_endpoint = endpoint, 
            api_key=subscription_key,  
            api_version="2024-05-01-preview"
        )
    
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        for task_name, json_path in TASKS.items():
            json_file = os.path.join(data_root, json_path)
            data_list = [json.loads(item.strip()) for item in open(json_file).readlines()]

            for data in data_list:
                question_id = data["question_id"]
                image_path = os.path.join(data_root, data['image'])
                assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
                                
                data_dict[question_id] = {
                    # required fields for data loading
                    "image_path": image_path,
                    # required fields for evaluation
                    "ground_truth": data["answer"],
                    "task_type": task_name,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}\nAnswer the question using a single word or phrase.'
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> str:
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
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            task_type = meta_data["task_type"]
            
            score = max([
                self.relaxed_correctness(ann, data["prediction"].strip(), meta_data["question"])
                for ann in ground_truth
            ])

            samples[task_type].append(score)
            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "score": score,
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
    
    # https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
    def relaxed_correctness(self,
                            target: str,
                            prediction: str,
                            question: str,
                            max_relative_change: float = 0.05) -> bool:
        """Calculates relaxed correctness.

        The correctness tolerates certain error ratio defined by max_relative_change.
        See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
        “Following Methani et al. (2020), we use a relaxed accuracy measure for the
        numeric answers to allow a minor inaccuracy that may result from the automatic
        data extraction process. We consider an answer to be correct if it is within
        5% of the gold answer. For non-numeric answers, we still need an exact match
        to consider an answer to be correct.”

        Args:
        target: Target string.
        prediction: Predicted string.
        max_relative_change: Maximum relative change.

        Returns:
        Whether the prediction was correct given the specified tolerance.
        """

        def _to_float(text: str) -> Optional[float]:
            try:
                if text.endswith('%'):
                    # Convert percentages to floats.
                    return float(text.rstrip('%')) / 100.0
                else:
                    return float(text)
            except ValueError:
                return None

        prediction_float = _to_float(prediction)
        target_float = _to_float(target)
        if prediction_float is not None and target_float:
            relative_change = abs(prediction_float -
                                target_float) / abs(target_float)
            relative_change1 = abs(prediction_float -
                                target_float/100) / abs(target_float/100)
            relative_change2 = abs(prediction_float/100 -
                                target_float) / abs(target_float)
            return relative_change <= max_relative_change or relative_change1 <= max_relative_change or relative_change2 <= max_relative_change
        else:
            return prediction.lower() == target.lower() or 'true' in self.match(target, prediction, question).lower()
        
    
    def build_prompt(self, gt, prediction, question):
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
            "Note that answers that are not exactly the same but have the same meaning or accurately answer the question should still be considered correct."
            "Please output according to the specified template."
            "Output template: True means correct, False means incorrect. Only output True or False.\n"
            "Example 1: \n"
            "Question: Which colored bar trumps all the bars?, Ground Truth: Dark Blue, Answer: Blue, Your output: False\n"
            "Example 2: \n"
            "Question: Which two values are same in the upper graph?, Ground Truth: [77, 77], Answer: 77, Your output: True\n"
            "Example 3: \n"
            "Question: What's the age strucutre in 2019 for 0-14 and 15-64?, Ground Truth: [42.47, 54.91], Answer: 42.47, Your output: False\n"
            "Example 4: \n"
            "Question: Which animal has the least cost of keeping?, Ground Truth: Rabbit**, Answer: Rabbit, Your output: True\n"
            "Example 5: \n"
            "Question: How many years has number of visitor below 1000?, Ground Truth: 3, Answer: 2, Your output: False\n"
            "Example 6: \n"
            "Question: Does the graph increase or decrease?, Ground Truth: increasing, Answer: Increase, Your output: True\n"
            "Question: {}, Ground Truth: {}, Answer: {}, Your output: "
        )
        return tmpl.format(question, gt, prediction)


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


    def match(self, gt, prediction, question):
        prompt = self.build_prompt(gt, prediction, question)
        retry_limit = 10
        
        for retry in range(retry_limit):
            try:
                extraction = self.get_chat_response(prompt, patience=10)
                return extraction
            except Exception as e:
                time.sleep(1)
        return 'False'

