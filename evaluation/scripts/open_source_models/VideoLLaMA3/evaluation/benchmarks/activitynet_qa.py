import json
import os
import re
import ast
import time
import copy
import random
import traceback
from typing import Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from openai import AzureOpenAI
from tqdm import tqdm

from .base import BaseVideoEvalDataset

endpoint = os.getenv("ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


check_template = [
    {
        "role": "system",
        "content":
            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer."
    },
    {
        "role": "user",
        "content":
            "Please evaluate the following video-based question-answer pair:\n\n"
            "Question: {question}\n"
            "Correct Answer: {answer}\n"
            "Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."
    }
]


def prompt_gpt(client, question, answer, pred, messages_template):
    messages = copy.deepcopy(messages_template)
    messages[1]["content"] = messages[1]["content"].format(question=question, answer=answer, pred=pred)
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
            response_message = completion.choices[0].message.content
            response_message = ast.literal_eval(response_message)
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

    return response_message


class ActivitynetQADataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "oqa"

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
        question_idx = 0

        # The question types are come from https://github.com/MILVLG/activitynet-qa/blob/master/evaluation/eval.py
        type_mapping = {
            0: "Motion",
            1: "Spatial Relation",
            2: "Temporal Relation",
            3: "Yes/No",
            4: "Color",
            5: "Object",
            6: "Location",
            7: "Number",
            8: "Other",
        }

        video_folder = os.path.join(data_root, "all_test")

        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

        records = json.load(open(os.path.join(data_root, "test_q.json")))
        answers = json.load(open(os.path.join(data_root, "test_a.json")))

        answer_mapping = {ele["question_id"]: {"answer": ele["answer"], "type": ele["type"]} for ele in answers}

        for data in records:
            # check if the video file exists
            video_name = "v_" + str(data["video_name"])
            video_path = None
            for video_format in video_formats:
                if os.path.exists(os.path.join(video_folder, video_name + video_format)):
                    video_path = os.path.join(video_folder, video_name + video_format)
                    break

            assert video_path is not None, f"Video file {video_name} not found in {video_folder}"

            data_dict[idx] = {
                # required fields for data loading
                "video_path": video_path,
                "start_time": None,
                "end_time": None,
                # required fields for evaluation
                "task_type": answer_mapping[data["question_id"]]["type"],
                "ground_truth": answer_mapping[data["question_id"]]["answer"],
                # custom fields for instruction generation and post processing
                "question": data["question"],
            }
            idx += 1

        # NOTE: debug string
        # select 10 samples from each type of questions
        # selected_data_dict = {}
        # selected_data_dict.update(dict(random.sample(data_dict.items(), 10)))
        
        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]

        # generate open-ended qa instruction
        instruction = f"Question: {question}\nPlease watch the video and answer the question clearly and concisely.\n"

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        question = self.data_dict[data_id]["question"]

        return response

    def evaluate(self, results):

        # For openai api evaluation, setting num_workers=2 is recommended
        def mt_process(func, lines, num_workers=2):
            # multi-threading processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                new_lines = list(tqdm(executor.map(func, lines), total=len(lines)))
            return new_lines

        def process_func(res):
            data = self.data_dict[res["data_id"]]
            check_dict = prompt_gpt(self.client, data["question"], data["ground_truth"], res["prediction"], check_template)
            res["score"] = float(check_dict["score"])
            res["match"] = int(check_dict["pred"] == "yes")

            return res

        total_results = mt_process(process_func, results)

        metrics, infos = {}, {}
        metrics["total"] = sum([res["match"] for res in total_results]) * 100 / len(total_results)
        metrics["total_score"] = sum([res["score"] for res in total_results]) / len(total_results)

        return metrics, infos
