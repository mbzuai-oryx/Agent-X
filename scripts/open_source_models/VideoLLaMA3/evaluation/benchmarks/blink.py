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


TASKS = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


class BLINKDataset(BaseImageEvalDataset):

    TASK_TYPES: List[str] = [task_type for task_type in TASKS]
    BENCHMARK_TYPE: str = "mcqa"
    
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
        
        self.client = AzureOpenAI(
            azure_endpoint = endpoint, 
            api_key=subscription_key,  
            api_version="2024-05-01-preview"
        )
    
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        self.set = "test"

        for task_type in TASKS:
            data_list = load_dataset(data_root, task_type)[self.set]

            for data in data_list:
                question_id = data["idx"]
                data_dict[question_id] = {
                    # required fields for data loading
                    "image_path": [data[f"image_{img_id}"] for img_id in range(1,5) if data[f"image_{img_id}"] is not None],
                    # required fields for evaluation
                    "task_type": task_type,
                    "ground_truth": data["answer"],
                    # custom fields for instruction generation and post processing
                    "question": data["prompt"],
                    "choices": data["choices"],
                    
                }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        # tags = "<image>\n"*len(meta_data["image_path"])
        instruction = f'{question}\nAnswer with the option\'s letter from the given choices directly.'
        return instruction
    
    def evaluate(self, results: List[Dict[str, Any]]):
        if self.set == "test":
            result_json = {}
            for data in results:
                result_json[data["data_id"]] = data["prediction"]
            return {}, result_json
        
        return super().evaluate(results)

    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        meta_data = self.data_dict[data_id]
        choices = meta_data["choices"]
        question = meta_data["question"]
        all_choices = ['(A)', '(B)', '(C)', '(D)', '(E)'][:len(choices)]
        intersect = list(set(all_choices).intersection(set(response.split())))
        intersect_last = list(set(all_choices).intersection(set(response.split('\n\n')[-1].split())))
        if response in ["A", "B", "C", "D", "E"]:
            prediction = "(" + response + ")"
        elif response in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            prediction = response
        elif (len(intersect) != 1 and len(intersect_last) != 1) or len(intersect) < 1:
            cs = ['(A)', '(B)', '(C)', '(D)', '(E)']
            options = '\n'.join([f'{cs[i]} {choices[i]}' for i in range(len(choices))])
            extracted_answer = self.match_multiple_choice(f"{question}\nSelect from the following choices", options, response)
            prediction = extracted_answer
        else:
            if len(intersect_last) == 1:
                intersect = intersect_last
                response = response.split('\n\n')[-1]
            prediction = intersect[0]
        return prediction
    
    
    def build_prompt(self, question, options, prediction):
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
            "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
            "You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. "
            "If the answer says things like refuse to answer, I'm sorry cannot help, etc., output (Z)"
            "If the meaning of all options are significantly different from the answer, or the answer does not select any option, output (Z)"\
            "Your should output one of the choices, (A),(B),(C),(D),(E) (if they are valid options), or (Z)\n"
            "Example 1: \n"
            "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: Point B, where the child is sitting, is closer to the camera.\nYour output: (B)\n"
            "Example 2: \n"
            "Question: Which point is closer to the camera?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer: I'm sorry, but I can't assist with that request.\nYour output: (Z)\n"
            "Example 3: \n"
            "Question: Which point is corresponding to the reference point?\nSelect from the following choices.\nOptions: (A) Point A\n(B) Point B\n(Z) Failed\nAnswer:The reference point (REF) on the first image is at the tip of the pot, which is the part used to Poke if the pots were used for that action. Looking at the second image, we need to find the part of the object that would correspond to poking.\n(A) Point A is at the tip of the spoon's handle, which is not used for poking.\n(B) Point B is at the bottom of the spoon, which is not used for poking.\n(C) Point C is on the side of the pspoonot, which is not used for poking.\n(D) Point D is at the tip of the spoon, which is not used for poking.\n\nTherefore, there is no correct answer in the choices\nYour output: (Z)\n"
            "Example 4: \n"
            "Question: {}?\nOptions: {}\n(Z) Failed\nAnswer: {}\nYour output: "
        )
        return tmpl.format(question, options, prediction)


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


    def match_multiple_choice(self, question, options, prediction):
        prompt = self.build_prompt(question, options, prediction)
        retry_limit = 10
        
        for retry in range(retry_limit):
            try:
                extraction = self.get_chat_response(prompt, patience=10)
                return extraction
            except Exception as e:
                time.sleep(1)
        return '(Z) Failed to get multiple choice'
