"""VideoLLaMA3 model configuration."""

import importlib.util
import os.path as osp
from typing import Optional, Dict, Any

from transformers import AutoConfig, AutoModel, PretrainedConfig, Qwen2Config

try:
    from .configuration_videollama3_encoder import Videollama3VisionEncoderConfig
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_videollama3_encoder",
        osp.join(osp.dirname(__file__), "configuration_videollama3_encoder.py"),
    )
    configuration_videollama3_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_videollama3_encoder)
    Videollama3VisionEncoderConfig = getattr(
        configuration_videollama3_encoder,
        "Videollama3VisionEncoderConfig",
    )

try:
    from .modeling_videollama3_encoder import Videollama3VisionEncoderModel
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "modeling_videollama3_encoder",
        osp.join(osp.dirname(__file__), "modeling_videollama3_encoder.py"),
    )
    modeling_videollama3_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling_videollama3_encoder)
    Videollama3VisionEncoderModel = getattr(
        modeling_videollama3_encoder,
        "Videollama3VisionEncoderModel",
    )

AutoConfig.register("videollama3_vision_encoder", Videollama3VisionEncoderConfig)
AutoModel.register(Videollama3VisionEncoderConfig, Videollama3VisionEncoderModel)


class Videollama3Qwen2Config(Qwen2Config):

    model_type = "videollama3_qwen2"
    sub_configs = {"vision_encoder_config": Videollama3VisionEncoderConfig}

    def __init__(
        self,
        vision_encoder: Optional[str] = None,
        vision_encoder_config: Dict[str, Any] = {},
        mm_projector_type: str = "mlp2x_gelu",
        use_token_compression: bool = True,
        image_token_index: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "videollama3_qwen2"

        self.vision_encoder = vision_encoder
        if vision_encoder_config is not None and not isinstance(vision_encoder_config, PretrainedConfig):
            vision_encoder_config = Videollama3VisionEncoderConfig(**vision_encoder_config)
        self.vision_encoder_config = vision_encoder_config

        self.mm_projector_type = mm_projector_type
        self.use_token_compression = use_token_compression
        self.image_token_index = image_token_index
