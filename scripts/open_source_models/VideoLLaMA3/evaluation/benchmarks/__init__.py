from .activitynet_qa import ActivitynetQADataset
from .base import BaseEvalDataset
from .charades_sta import CharadesSTADataset
from .egoschema import EgoSchemaDataset
from .longvideobench import LongVideoBenchDataset
from .lvbench import LVBenchDataset
from .mlvu import MLVUDataset
from .mmvu import MMVUDataset
from .mvbench import MVBenchDataset
from .nextqa import NextQADataset
from .perception_test import PerceptionTestDataset
from .tempcompass import TempCompassDataset
from .videomme import VideoMMEDataset

from .ai2d import AI2DDataset
from .chartqa import ChartQADataset
from .docvqa import DocVQADataset
from .mathvista import MathVistaDataset
from .mmmu import MMMUDataset
from .ocrbench import OCRBenchDataset
from .gqa import GQADataset
from .mmmupro import MMMUProDataset
from .realworldqa import RealWorldQADataset
from .blink import BLINKDataset
from .mme import MMEDataset
from .infovqa import InfoVQADataset
from .mathverse import MathVerseDataset
from .mathvision import MathVisionDataset

DATASET_REGISTRY = {
    "videomme": VideoMMEDataset,
    "mmvu": MMVUDataset,
    "mvbench": MVBenchDataset,
    "egoschema": EgoSchemaDataset,
    "perception_test": PerceptionTestDataset,
    "activitynet_qa": ActivitynetQADataset,
    "mlvu": MLVUDataset,
    "longvideobench": LongVideoBenchDataset,
    "lvbench": LVBenchDataset,
    "tempcompass": TempCompassDataset,
    "nextqa": NextQADataset,
    "charades_sta": CharadesSTADataset,
    "AI2D": AI2DDataset,
    "ChartQA": ChartQADataset,
    "DocVQA": DocVQADataset,
    "MathVista": MathVistaDataset,
    "MMMU": MMMUDataset,
    "OCRBench": OCRBenchDataset,
    "GQA": GQADataset,
    "MMMU_Pro": MMMUProDataset,
    "RealWorldQA": RealWorldQADataset,
    "BLINK": BLINKDataset,
    "MME": MMEDataset,
    "InfoVQA": InfoVQADataset,
    "MathVerse": MathVerseDataset,
    "MathVision": MathVisionDataset,
}


def build_dataset(benchmark_name: str, **kwargs) -> BaseEvalDataset:
    assert benchmark_name in DATASET_REGISTRY, f"Unknown benchmark: {benchmark_name}, available: {DATASET_REGISTRY.keys()}"
    return DATASET_REGISTRY[benchmark_name](**kwargs)
