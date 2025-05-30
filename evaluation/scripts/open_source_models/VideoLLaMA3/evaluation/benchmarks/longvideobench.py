import json
import os
import re
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import pyarrow.parquet as pq
import pysubs2

from .base import BaseVideoEvalDataset
from videollama3.mm_utils import load_video


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(':')
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, 
                                 starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration
                
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]

            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
                if frame_timestamp <= subtitle_timestamp:
                    # print("frame:", frame_timestamp)
                    interleaved_list.append(frame)
                    cur_i += 1
                else:
                    break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        #
        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)
        
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        # print(frame_timestamp)
        interleaved_list.append(frame)
        
    return interleaved_list


class LongVideoBenchDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"

    def load_data(self, data_root: str) -> Dict[int, Any]:
        json_file = os.path.join(data_root, "lvb_val.json")
        records = json.load(open(json_file, "r"))

        video_folder = os.path.join(data_root, "videos")
        subtitle_folder = os.path.join(data_root, "subtitles")
        data_dict = {}
        idx = 0

        lengths = []
        for record in records:
            video_path = os.path.join(video_folder, record["video_path"])
            assert os.path.exists(video_path), f"Cannot find the video file: {video_path}"

            subtitle_path = os.path.join(subtitle_folder, record["subtitle_path"])
            subtitle = json.load(open(subtitle_path, 'r'))

            meta_data = {
                # required fields for data loading
                "video_path": video_path,
                "start_time": None,
                "end_time": None,
                # required fields for evaluation
                "task_type": record["question_category"],
                "level": record["level"],
                "question_category": record["question_category"],
                "topic_category": record["topic_category"],
                "duration_group": record["duration_group"],
                "ground_truth": chr(ord("A") + record["correct_choice"]),
                # custom fields for instruction generation and post processing
                "question": record["question"],
                "options": list(record["candidates"]),
                "subtitle": subtitle,
                "starting_timestamp_for_subtitles": record["starting_timestamp_for_subtitles"],
                "duration": record["duration"],
            }

            data_dict[idx] = meta_data

            idx += 1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video_text_interleave_list, timestamps: Any) -> Dict[str, str]:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]

        instruction = []

        instruction += ["Question: " + question]
        instruction += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(options)]
        instruction += ["Answer with the option's letter from the given choices directly."]
        instruction = "\n".join(instruction)

        video_text_interleave_list.append(instruction.strip())

        # make conversation
        content_list = []
        cur_frames = []
        frame_idx = 0
        for ele in video_text_interleave_list:
            if isinstance(ele, str):
                if len(cur_frames) > 0:
                    content_list.append(
                        {
                            "type": "video",
                            "num_frames": len(cur_frames),
                            "timestamps": timestamps[frame_idx-len(cur_frames):frame_idx]
                        }
                    )
                    cur_frames = []
                content_list.append({"type": "text", "text": ele})

            elif isinstance(ele, np.ndarray):
                cur_frames.append(ele)
                frame_idx += 1

        conversation = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        return conversation

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        aggregated_data = self._aggregated_data_list[idx]

        try:
            frames, timestamps = self.processor.load_video(
                aggregated_data["video_path"],
                start_time=aggregated_data["start_time"],
                end_time=aggregated_data["end_time"],
                precise_time=True,
                fps=self.fps,
                max_frames=self.max_frames,
            )
            image_inputs = self.processor.process_images(
                [frames],
                merge_size=2,
                return_tensors="pt"
            )
        except:
            traceback.print_exc()
            print(f"Failed to load video: {aggregated_data}")
            exit()

        text_inputs = []
        for data_id in aggregated_data["data_ids"]:
            meta_data = self.data_dict[data_id]
            subtitle = meta_data["subtitle"]
            video_text_interleave_list = insert_subtitles_into_frames(frames, timestamps, subtitle, meta_data["starting_timestamp_for_subtitles"], meta_data["duration"])
            conversation = self.generate_instruction(data_id, video_text_interleave_list, timestamps)
            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            text_inputs.append(
                self.processor.process_text(
                    prompt,
                    image_inputs,
                    padding=False,
                    padding_side=None,
                    return_tensors="pt"
                )
            )

        data = {
            "data_ids": aggregated_data["data_ids"],
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
        }

        return data

    def process_response(self, data_id: Union[int, str], response: str) -> str:
        options = self.data_dict[data_id]["options"]
        # options = [re.findall('[A-D]\. (.*).', x)[0] for x in self.data_dict[data_id]["options"]]
        letters = [chr(ord("A") + i) for i in range(len(options))]
        digit2word = {
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine',
            '0': 'zero',
        }

        response = response.replace("answer", "")
        response = response.replace("Answer", "")
        pred_answer = re.findall(f"[\(\ \[]*([{letters[0]}-{letters[-1]}])[\)\.\ \]]*", response)

        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                # Arabic numerals -> English words
                opt2 = opt
                if opt in digit2word:
                    opt2 = digit2word[opt]
                if opt.lower() in response.lower() or opt2.lower() in response.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        prediction = letters[pred_idx]

        prediction = prediction.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is"
            "The correct option is",
            "Best answer:"
            "Best option:",
        ]
        for answer_prefix in answer_prefixes:
            prediction = prediction.replace(answer_prefix, "")

        if len(prediction.split()) > 10 and not re.search(f"[{letters[0]}-{letters[-1]}]", prediction):
            raise ValueError(f"Cannot find the answer in the options: {prediction}")
        matches = re.search(rf'[{letters[0]}-{letters[-1]}]', prediction)
        if matches is None:
            raise ValueError(f"Cannot find the answer in the options: {prediction}")
        prediction = matches[0]

        return prediction

    def evaluate(self, results: List[Dict[str, Any]]) -> (Dict[str, Dict[str, float]], Dict[str, List[Dict[str, Any]]]):

        l1_results = [x for x in results if self.data_dict[x["data_id"]]["level"] == "L1-Perception"]
        l2_results = [x for x in results if self.data_dict[x["data_id"]]["level"] == "L2-Relation"]

        metrics, infos = {}, {}
        metrics["Overall"], infos["Overall"] = super().evaluate(results)
        metrics["L1-Perception"], infos["L1-Perception"] = super().evaluate(l1_results)
        metrics["L2-Relation"], infos["L2-Relation"] = super().evaluate(l2_results)

        return metrics, infos
