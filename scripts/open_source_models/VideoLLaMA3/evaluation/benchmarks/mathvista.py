import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
from collections import defaultdict
from Levenshtein import distance
from openai import AzureOpenAI

from .base import BaseImageEvalDataset, filter_metadata


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

endpoint = os.getenv("ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

class MathVistaDataset(BaseImageEvalDataset):

    # TASK_TYPES: List[str] = [task_type for task_type in TASKS]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AzureOpenAI(
            azure_endpoint = endpoint, 
            api_key=subscription_key,  
            api_version="2024-05-01-preview"
        )
    
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        json_file = os.path.join(data_root, "testmini.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)
        query_file = os.path.join(data_root, "query.json")
        with open(query_file, "r") as f:
            query_list = json.load(f)

        for question_id in data_list:
            data = data_list[question_id]
            image_path = os.path.join(data_root, data['image'])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
                            
            data_dict[question_id] = {
                # required fields for data loading
                "image_path": image_path,
                # required fields for evaluation
                "task_type": data["question_type"],
                "ground_truth": data["answer"],
                # custom fields for instruction generation and post processing
                "answer_type": data["answer_type"],
                "precision": data["precision"],
                "question": query_list[question_id],
                "options": data["choices"]
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}'
        return instruction

    def get_most_similar(self, prediction, choices):
        """
        Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
        """
        distances = [distance(prediction, choice) for choice in choices]
        ind = distances.index(min(distances))
        return choices[ind]
    
    def get_chat_response(self, promot, n=1, patience=10000000, sleep_time=0):
        messages = [
            {"role": "user", "content": promot},
        ]
        # print("I am here")
        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(
                    model=deployment,
                    messages = messages,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
                else:
                    prediction = [choice.text.strip() for choice in response.choices]
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
    
    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        task_type = self.data_dict[data_id]["task_type"]
        answer_type = self.data_dict[data_id]["answer_type"]
        question = self.data_dict[data_id]["question"]
        options = self.data_dict[data_id]["options"]
        
        if response == "":
            return ""

        if task_type == 'multi_choice' and response in options:
            return response

        if answer_type == "integer":
            try:
                response = int(response)
                return str(response)
            except:
                pass
        
        if answer_type == "float":
            try:
                response = str(float(response))
                return response
            except:
                pass
        
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                response = result.group(1)
                return response
        except:
            pass
            
        # general extraction
        try:
            full_prompt = f"{demo_prompt.strip()}\n\n{question}\n\n{response}\n\nExtracted answer: "
            response = self.get_chat_response(full_prompt, patience=10)
            return response
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {data_id}")
        
        return ""
    
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
            answer_type = meta_data["answer_type"]
            precision = meta_data["precision"]
            options = meta_data["options"]
            ground_truth = meta_data["ground_truth"]
            response = data["prediction"]
            if task_type == 'multi_choice':
                # make sure the extraction is a string
                if isinstance(response, str):
                    response = response.strip()
                else:
                    try:
                        response = str(response)
                    except:
                        response = ""

                # extract "A" from "(A) text"
                letter = re.findall(r'\(([a-zA-Z])\)', response)
                if len(letter) > 0:
                    response = letter[0].upper()

                choices = [chr(ord('A') + i) for i in range(len(options))]

                if response in choices:
                    # convert option letter to text, e.g. "A" -> "text"
                    ind = choices.index(response)
                    response = options[ind]
                else:
                    # select the most similar option
                    response = self.get_most_similar(response, options)
                assert response in options

            elif answer_type == 'integer':
                try:
                    response = str(int(float(response)))
                except:
                    response = None

            elif answer_type == 'float':
                try:
                    response = str(round(float(response), int(precision)))
                except:
                    response = None

            elif answer_type == 'list':
                try:
                    response = str(response)
                except:
                    response = None

            correct = False
            try:
                if response == ground_truth:
                    correct = True
            except Exception as e:
                print(e)
                
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
    