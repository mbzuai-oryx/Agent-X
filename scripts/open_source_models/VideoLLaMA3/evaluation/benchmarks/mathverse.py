import json
import os
import re
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union
from collections import defaultdict
from openai import AzureOpenAI
from tqdm import tqdm

from .base import BaseImageEvalDataset, filter_metadata


demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""

demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """

endpoint = os.getenv("ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


class MathVerseDataset(BaseImageEvalDataset):

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

        for data in data_list:
            if data["problem_version"] != "Vision Only":
                continue
            question_id = data["sample_index"]
            image_path = os.path.join(data_root, "images", data['image'])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
                            
            data_dict[question_id] = {
                # required fields for data loading
                "image_path": image_path,
                # required fields for evaluation
                "task_type": data["question_type"],
                "ground_truth": data["answer"],
                # custom fields for instruction generation and post processing
                "question": data["query_cot"] # data["query_wo"], data["query_cot"]
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f'{question}'
        return instruction
    
    def get_chat_response(self, promot, n=1, patience=10000000, sleep_time=0, max_tokens=256):
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
                    temperature=0,
                    max_tokens=max_tokens,
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
    
    def create_extract_prompt(self, demo_prompt, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"Model response: '{response}'\nExtracted Answer: "
        full_prompt = f"{demo_prompt}\n\n{test_prompt}"
        return full_prompt

    def create_match_prompt(self, demo_prompt, question, answer, extraction):
        demo_prompt = demo_prompt.strip()
        full_prompt = demo_prompt.format(question=question, gt=answer, extraction=extraction)
        return full_prompt
    
    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        if not response:
            return ""

        # general extraction
        try:
            full_prompt = self.create_extract_prompt(demo_prompt_extract, response)
            extraction = self.get_chat_response(full_prompt)
            return extraction
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
        
        for data in tqdm(results):
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            task_type = meta_data["task_type"]
            ground_truth = meta_data["ground_truth"]
            question = meta_data["question"]
            response = data["prediction"]
            
            correct = False
            quick_match = False
            if quick_match:
                if response == ground_truth:
                    correct = True
            else:
                try:
                    full_prompt = self.create_match_prompt(demo_prompt_score, question, ground_truth, response)
                    while True:
                        extraction = self.get_chat_response(full_prompt, max_tokens=8)
                        judgement = extraction.replace("Judgement:", "").strip()
                        if judgement.strip() in ["0", "1"]:
                            if int(judgement) == 1:
                                correct = True
                            break
                except Exception as e:
                    print(e)
                    print(f"Error in matching answer")
                
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
    