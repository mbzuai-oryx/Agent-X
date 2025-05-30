# Adopted from https://github.com/haotian-liu/LLaVA.
# Below is the original copyright:
# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch VideoLLaMA3 model."""

import importlib.util
import os.path as osp
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers import AutoModel, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_videollama3 import Videollama3Qwen2Config
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "configuration_videollama3",
        osp.join(osp.dirname(__file__), "configuration_videollama3.py"),
    )
    configuration_videollama3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration_videollama3)
    Videollama3Qwen2Config = getattr(
        configuration_videollama3,
        "Videollama3Qwen2Config",
    )


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


def build_vision_projector(config, delay_load=False, **kwargs):
    # videollama3 projector only support image-wise operation now, i.e., prohibit the temporal aggregation
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(config, projector_type)
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class MlpGeluProjector(nn.Module):

    def __init__(self, config, projector_type):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(mlp_depth, config.vision_encoder_config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.readout(x)
        return x


class Videollama3MetaModel:

    def __init__(self, config):
        super(Videollama3MetaModel, self).__init__(config)
        if config.vision_encoder is not None:
            self.vision_encoder = AutoModel.from_pretrained(
                config.vision_encoder,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
            self.config.vision_encoder_config = self.vision_encoder.config
            self.config.vision_encoder = None
        elif config.vision_encoder_config is not None:
            self.vision_encoder = AutoModel.from_config(
                self.config.vision_encoder_config,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.dtype,
            )
        else:
            raise ValueError("Vision encoder is not provided in config")
        self.mm_projector = build_vision_projector(config)

    def get_vision_encoder(self):
        return self.vision_encoder

    def get_mm_projector(self):
        return self.mm_projector


class Videollama3Qwen2Model(Videollama3MetaModel, Qwen2Model):

    config_class = Videollama3Qwen2Config

    def __init__(self, config: Videollama3Qwen2Config):
        super(Videollama3Qwen2Model, self).__init__(config)


class Videollama3MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ) -> torch.FloatTensor:
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        mm_features = self.get_model().mm_projector(mm_features)
        return mm_features

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full((num_patches, ), modal != "text", dtype=torch.bool, device=mm_features.device)
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def _maybe_truncate_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_token_index).sum():
            return mm_features, compression_mask

        truncation_mask = []
        for num_patches, modal in zip(batched_num_patches, modals):
            if modal == "text":
                truncation_mask.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))
            else:
                truncation_mask.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

        seq_end_indices = torch.nonzero(position_ids == 0)[:, 0]
        seq_end_indices = seq_end_indices[seq_end_indices > 0].tolist()+ [len(input_ids)]
        seq_start_indices = [0] + seq_end_indices[:-1]
        num_visual_tokens = [
            input_ids[start:end].eq(self.config.image_token_index).sum()
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]

        for n, mask in zip(num_visual_tokens, truncation_mask):
            if len(mask) > 0:
                mask[n:] = False
        truncation_mask = torch.cat(truncation_mask)

        return mm_features[truncation_mask], compression_mask[truncation_mask]

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, num_patches, grid_size, merge_size, modal in zip(
            batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            t, h, w = grid_size
            if modal == "image" or (modal == "video" and t == 1):
                compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))

            elif modal == "video":
                # NOTE: video token compressor
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = pixel_diff > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                # mask[padding_ids, torch.randperm(min_tokens)] = 1
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

            else:
                # in case of psuedo image
                compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))

        return torch.cat(compression_masks)

    def _compress_visual_tokens(
        self,
        compression_mask: torch.BoolTensor,
        mm_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        mm_features = mm_features[compression_mask]
        image_selected = (input_ids == self.config.image_token_index)

        text_masks = torch.logical_not(image_selected)
        text_masks[image_selected] = compression_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            # FIXME: assume the first position_id is always 0
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([torch.arange(end - start, device=input_ids.device) for start, end in zip(pos_start, pos_end)])

        return mm_features, input_ids, attention_mask, position_ids, labels

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
    ):
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels

        # 1. flatten text inputs
        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens
        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
        mm_features = self.encode_images(pixel_values, grid_sizes, merge_sizes)
        mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)

        compression_mask = self._get_compression_mask(
            pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
        )
        mm_features, compression_mask = self._maybe_truncate_visual_tokens(
            mm_features, compression_mask, batched_num_patches, modals, input_ids, position_ids
        )

        # 3. compress visual tokens
        if self.config.use_token_compression:
            assert B == 1, "Token compression is only supported for batch_size=1"
            mm_features, input_ids, attention_mask, labels, position_ids = self._compress_visual_tokens(
                compression_mask, mm_features, input_ids, attention_mask, labels, position_ids
            )

        # 4. embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        # 5. replace multimodal tokens with features
        image_selected = (input_ids == self.config.image_token_index)
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features   

        # 6. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels


class Videollama3Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama3MetaForCausalLM):

    config_class = Videollama3Qwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Videollama3Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # NOTE: arguments are copied from transformers==4.46.3
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        # multimodal inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
