import json
import os
import re
import time
import random
import traceback
from typing import Dict, Any, Union, List
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from openai import AzureOpenAI
from tqdm import tqdm

from .base import BaseVideoEvalDataset

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo-0613")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


caption_evaluation_prompt = """
You will receive a video description and a multi-choice question. Your task is to choose the correct answer and briefly explain the reason why you choose the answer. \
If none of the choice candidates are correct or the video description lacks enough information to answer the question, just answer "None". \
Please organize your response in this format:
```
Reasoning: [Your reason to obtain the answer]
Answer: [Your answer]
```

Here are some examples of video description, multi-choice question and the expected answer:
```
Video Description: A person is palying football.
Multi-Choice Question:
What is the person doing in the video?
Information A: {'subject': 'person', 'action': 'cooking'}
Information B: {'subject': 'person', 'action': 'palying football'} 
Information C: {'subject': 'person', 'action': 'playing basketball'} 
Information D: {'subject': 'person', 'action': 'reading book'} 
Reasoning: The video description mentions that the person is playing football.
Answer: B

Video Description: A bird is flying clockwise.
Multi-Choice Question:
In which direction is the bird flying?
Information A: {'subject': 'bird', 'action': 'backwark'}
Information B: {'subject': 'bird', 'action': 'counter-clockwise'}
Information C: {'subject': 'bird', 'action': 'clockwise'}
Information D: {'subject': 'bird', 'action': 'downward'}
Reasoning: The video description mentions that the bird is flying clockwise
Answer: C

Video Description: An air balloon is inflating.
Multi-Choice Question:
What is happening to the air balloon?
Information A: {'subject': 'air ballon', 'action': 'exploding'}
Information B: {'subject': 'air ballon', 'action': 'getting smaller'}
Information C: {'subject': 'air ballon', 'action': 'flying'}
Reasoning: The video description mentions that the air balloon is inflating, while none of the coices can be explained as inflating.
Answer: None
```
"""


def prompt_gpt(client, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant for question answering."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    while True:
        try:
            # Generate the completion  
            completion = client.chat.completions.create(  
                model=deployment,
                messages=messages,
                max_tokens=800,  
                temperature=0.7,  
                top_p=0.95,  
                frequency_penalty=0,  
                presence_penalty=0,
                stop=None,  
                stream=False
            )
        except:
            traceback.print_exc()
            # if maxtry <= 0:
            #     eval_result = {"chatgpt-reasoning": None, "chatgpt-answer": None, "rating": -1, "token_count": None}
            #     return eval_result
            # maxtry -= 1
            print(f"Not success! retries remaining...")
            time.sleep(random.uniform(3, 5))
            continue
        break

    response_message = completion.choices[0].message.content

    # find A, B, C, D, or None
    prediction_answer = re.findall(r"Answer: [A-D]|Answer: None", response_message)

    if len(prediction_answer) == 0:
        return "None"
    else:
        return prediction_answer[0].split("Answer:")[1].strip()


class TempCompassDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"
    TASK_TYPES: List[str] = ["action", "direction", "speed", "order", "attribute_change"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize Azure OpenAI client with key-based authentication
        self.client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version="2024-05-01-preview",  
        )

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        for task_format in ["yes_no", "captioning", "caption_matching", "multi-choice"]:
            parquet_file = os.path.join(data_root, task_format, "test-00000-of-00001.parquet")
            video_folder = os.path.join(data_root, "videos")

            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            for data in df.itertuples():
                data_dict[idx] = {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, data.video_id + ".mp4"),
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": data.dim,
                    "question_format": task_format,
                    "ground_truth": data.answer,
                    # custom fields for instruction generation and post processing
                    "question": data.question,
                }
                idx += 1

        # NOTE: debug string
        # select 10 samples from four types of questions
        # selected_data_dict = {}
        # selected_data_dict.update(dict(random.sample({k: v for k, v in data_dict.items() if v["question_format"] == "yes_no"}.items(), 10)))
        # selected_data_dict.update(dict(random.sample({k: v for k, v in data_dict.items() if v["question_format"] == "captioning"}.items(), 10)))
        # selected_data_dict.update(dict(random.sample({k: v for k, v in data_dict.items() if v["question_format"] == "caption_matching"}.items(), 10)))
        # selected_data_dict.update(dict(random.sample({k: v for k, v in data_dict.items() if v["question_format"] == "multi-choice"}.items(), 10)))

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]

        question_format = meta_data["question_format"]

        if question_format == "yes_no":
            instruction = f"Question: {question}\nPlease answer with 'yes' or 'no' directly."
        elif question_format == "captioning":
            question = question.replace("Generated Caption:", "").strip()
            instruction = f"Question: {question}\nPlease answer with the caption directly."
        elif question_format == "caption_matching":
            instruction = f"Question: {question}\nPlease directly give the best option:"
        elif question_format == "multi-choice":
            instruction = f"Question: {question}\nPlease answer with the option\'s letter from the given choices directly and only give the best option."

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        meta_data = self.data_dict[data_id]

        question_format = meta_data["question_format"]
        question = meta_data["question"].replace("\nGenerated Caption:", "")
        answer = meta_data["ground_truth"]

        if question_format == "yes_no":
            assert "yes" in response.lower() or "no" in response.lower(), f"Invalid yes_no type response: {response}"
            if response.lower().startswith(answer):
                pred = answer
            else:
                pred = response
        elif question_format == "captioning":
            pred = response.lower()
        elif question_format == "caption_matching":
            pred = response.lower()
            gt = answer.split(':')[0].strip().lower()
            raw_gt = gt
            prefix_words = ['option', 'sentence', 'caption']
            for prefix_word in prefix_words:
                gt = gt.replace(prefix_word, '').strip()
                pred = pred.replace(prefix_word, '').strip()
            if pred.startswith(gt):
                pred = answer
            else:
                pred = response
        elif question_format == "multi-choice":
            pred = response.lower()
            gt = answer.split('.')[0].strip().lower()
            if pred.startswith(gt):
                pred = answer
            else:
                pred = response

        return pred

    def evaluate(self, results):

        def mt_process(func, lines, num_workers=16):
            # multi-process
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                new_lines = list(tqdm(executor.map(func, lines), total=len(lines)))
            new_lines = [line for line in new_lines if line is not None]
            return new_lines

        def process_func(res):
            data = self.data_dict[res["data_id"]]
            if data["question_format"] == "yes_no":
                return res
            elif data["question_format"] == "caption_matching":
                return res
            elif data["question_format"] == "multi-choice":
                return res
            elif data["question_format"] == "captioning":
                question = data["question"]
                answer = data["ground_truth"]
                response = res["prediction"]
                question = question.replace("\nGenerated Caption:", "").strip()
                prompt = f"""{caption_evaluation_prompt}\nVideo Description: {response}\nMulti-Choice Question:\n{question}\n"""
                pred = prompt_gpt(self.client, prompt).lower()
                gt = answer.split('.')[0].strip().lower()
                if pred.startswith(gt):
                    pred = answer
                res['prediction'] = pred
            return res

        total_results = mt_process(process_func, results)

        yes_no_results = [res for res in total_results if self.data_dict[res["data_id"]]["question_format"] == "yes_no"]
        captioning_results = [res for res in total_results if self.data_dict[res["data_id"]]["question_format"] == "captioning"]
        captioning_matching_results = [res for res in total_results if self.data_dict[res["data_id"]]["question_format"] == "caption_matching"]
        multi_choice_results = [res for res in total_results if self.data_dict[res["data_id"]]["question_format"] == "multi-choice"]

        metrics, infos = {}, {}
        metrics["yes_no"], infos["yes_no"] = super().evaluate(yes_no_results)
        metrics["captioning"], infos["captioning"] = super().evaluate(captioning_results)
        metrics["caption_matching"], infos["caption_matching"] = super().evaluate(captioning_matching_results)
        metrics["multi_choice"], infos["multi_choice"] = super().evaluate(multi_choice_results)
        metrics["total"], infos["total"] = super().evaluate(total_results)

        return metrics, infos
