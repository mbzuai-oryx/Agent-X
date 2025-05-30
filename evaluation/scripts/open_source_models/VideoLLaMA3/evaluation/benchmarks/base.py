import queue
import threading
import traceback
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from torch.utils.data import Dataset, DataLoader

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.mm_utils import load_video


def filter_metadata(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(data[key], (dict, list)):
                new_data[key] = filter_metadata(value)
            elif isinstance(data[key], (int, float, bool, str)):
                new_data[key] = value
        return new_data
    elif isinstance(data, list):
        new_data = []
        for item in data:
            if isinstance(item, (dict, list)):
                new_data.append(filter_metadata(item))
            elif isinstance(item, (int, float, bool, str)):
                new_data.append(item)
        return new_data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


class BaseEvalDataset(Dataset, metaclass=ABCMeta):

    BENCHMARK_TYPE: str = None
    TASK_TYPES: List[str] = None
    MODAL: str = None

    def __init__(
        self,
        data_root: str,
        processor: Callable,
        num_splits: int = 1,
        split_idx: int = 0,
        fps: int = 1,
        max_frames: int = 180,
    ) -> None:
        assert split_idx < num_splits, f"split_idx ({split_idx}) should be less than num_splits ({num_splits})"
        self.processor = processor
        self.data_dict = self.load_data(data_root)


    @property
    def n_samples(self) -> int:
        return sum([len(x["data_ids"]) for x in self._aggregated_data_list])

    def __len__(self) -> int:
        return len(self._aggregated_data_list)

    @abstractmethod
    def load_data(self, data_root) -> Dict[Union[int, str], Any]:
        """
        Load the dataset meta data.

        Args:
            data_root (str): path to the dataset.

        Returns:
            data_dict (Dict[Union[int, str], Any]): dataset meta data, with data_id as key.
            example:
            {
                0: {
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
                ...
            }
        """
        pass

    @abstractmethod
    def generate_instruction(self, data_id: Union[int, str]) -> Union[str, Dict[str, str]]:
        """
        Generate instruction(s) for model inference.

        Args:
            data_id (Union[int, str]): identifier of the data.

        Returns:
            instruction (Union[str, Dict[str, str]]): instruction(s) for model inference.
        """
        pass

    @abstractmethod
    def process_response(self, data_id: Union[int, str], response: str) -> Any:
        """
        Process the original model responses to desired format for evaluation and visualization.

        Args:
            data_id (Union[int, str]): identifier of the data.
            response (str): model response.

        Returns:
            result (Any): processed model response for evaluation.
        """
        pass

    def evaluate(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics according to predictions and ground-truths.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        assert self.BENCHMARK_TYPE is not None, "BENCHMARK_TYPE is not defined."
        if self.TASK_TYPES is None:
            warnings.warn("TASK_TYPES is not defined. It will be automatically inferred from metadata.")
        if self.BENCHMARK_TYPE == "mcqa":
            return self._eval_mcqa(results)
        elif self.BENCHMARK_TYPE == "oqa":
            return self._eval_oqa(results)
        elif self.BENCHMARK_TYPE == "temporal_grounding":
            return self._eval_temporal_grounding(results)
        else:
            raise NotImplementedError(f"Unsupported benchmark type: {self.BENCHMARK_TYPE}")

    def _eval_mcqa(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics for multiple-choice question answering tasks.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        if self.TASK_TYPES is None:
            samples = defaultdict(list)
        else:
            samples = {task_type: [] for task_type in self.TASK_TYPES}

        overall_samples = []
        infos = []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            ground_truth = meta_data["ground_truth"]
            task_type = meta_data["task_type"]
            matching = data["prediction"] == meta_data["ground_truth"]

            if isinstance(task_type, (list, tuple)):
                for t in task_type:
                    samples[t].append(int(matching))
            else:
                samples[task_type].append(int(matching))

            overall_samples.append(int(matching))

            infos.append(
                {
                    **data,
                    "ground_truth": ground_truth,
                    "matching": matching,
                    "task_type": task_type,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        task_types = samples.keys()
        metrics = {x: sum(samples[x]) / len(samples[x]) * 100 for x in task_types}

        # overall_samples = sum(samples.values(), [])
        overall_acc = sum(overall_samples) / len(overall_samples) * 100
        metrics["Overall"] = overall_acc

        infos = [metrics] + infos
        return metrics, infos

    def _eval_oqa(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        """
        Compute the evaluation metrics for open-ended question answering tasks.

        Args:
            results (List[Dict[str, Any]]): list of processed model responses.

        Returns:
            metrics (Dict[str, float]): evaluation metrics.
            infos (List[Dict[str, Any]]): evaluation information for visualization.
        """
        samples = []
        infos = []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            score = data["score"]

            samples.append(score)
            infos.append(
                {
                    **data,
                    "score": score,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        metrics = {"Overall": sum(samples) / len(samples)}

        infos = [metrics] + infos
        return metrics, infos

    def _eval_temporal_grounding(self, results: List[Dict[str, Any]]) -> (Dict[str, float], List[Dict[str, Any]]):
        ious, infos = [], []

        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            gt_interval = meta_data["ground_truth"]

            intersection = 0
            union = gt_interval[1] - gt_interval[0]
            for pred_interval in data["prediction"]:
                start_time, end_time = min(pred_interval), max(pred_interval)
                intersection += max(0, min(end_time, gt_interval[1]) - max(start_time, gt_interval[0]))
                union += end_time - start_time
            union = union - intersection
            iou = intersection / union

            ious.append(iou)
            infos.append(
                {
                    **data,
                    "ground_truth": gt_interval,
                    "iou": iou,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        metrics = {
            "mIoU": sum(ious) / len(ious) * 100,
        }
        for thred in [0.3, 0.5, 0.7]:
            metrics[f"R1@{thred}"] = sum(iou >= thred for iou in ious) / len(ious) * 100

        infos = [metrics] + infos
        return metrics, infos


class BaseImageEvalDataset(BaseEvalDataset):
    
    MODAL: str = "image"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            image_path = meta_data["image_path"]
            if image_path not in aggregated_data:
                aggregated_data[image_path] = {
                    "image_path": image_path,
                    "data_ids": [data_id],
                }
            else:
                aggregated_data[image_path]["data_ids"].append(data_id)

        aggregated_data_list = [x for _, x in aggregated_data.items()]
        self._aggregated_data_list = aggregated_data_list[kwargs["split_idx"]::kwargs["num_splits"]]
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        aggregated_data = self._aggregated_data_list[idx]
        
        try:
            images = self.processor.load_images(aggregated_data["image_path"])
            image_inputs = self.processor.process_images(images, merge_size=1, return_tensors="pt")
        except:
            traceback.print_exc()
            print(f"Failed to load image: {aggregated_data}")
            exit()
            
        text_inputs = []
        for data_id in aggregated_data["data_ids"]:
            instruction = self.generate_instruction(data_id)
            content = [{"type": "image"}] * (len(aggregated_data["image_path"]) if isinstance(aggregated_data["image_path"], list) else 1)
            conversation = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": content + [{"type": "text", "text": instruction}],
                }
            ]
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
        
        
class BaseVideoEvalDataset(BaseEvalDataset):
    
    MODAL: str = "video"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fps = kwargs["fps"]
        self.max_frames = kwargs["max_frames"]

        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            video_path = meta_data["video_path"]
            start_time = meta_data["start_time"]
            end_time = meta_data["end_time"]
            aggregated_data_id = f"{video_path}_{start_time}_{end_time}"
            if aggregated_data_id not in aggregated_data:
                aggregated_data[aggregated_data_id] = {
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "data_ids": [data_id],
                }
            else:
                aggregated_data[aggregated_data_id]["data_ids"].append(data_id)

        aggregated_data_list = [x for _, x in aggregated_data.items()]
        self._aggregated_data_list = aggregated_data_list[kwargs["split_idx"]::kwargs["num_splits"]]

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
            instruction = self.generate_instruction(data_id, timestamps)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "num_frames": len(timestamps),
                            "timestamps": timestamps,
                        },
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
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
