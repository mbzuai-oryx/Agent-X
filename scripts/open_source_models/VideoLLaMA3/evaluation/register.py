from transformers import AutoConfig

import sys
sys.path.append('./')

try:
    import videollama2
except ImportError:
    videollama2 = None

import videollama3


def model_init(model_path, max_visual_tokens=None, **kwargs):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    device_map = kwargs.get('device_map', {"": "cuda:0"})
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if max_visual_tokens is not None:
        processor.image_processor.max_tokens = max_visual_tokens
    return model, processor


def mm_infer(data_dict, model, tokenizer, modal='video', **kwargs):
    import torch
    from videollama3.mm_utils import KeywordsStoppingCriteria
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data_dict["input_ids"])

    data_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
    if "pixel_values" in data_dict:
        data_dict["pixel_values"] = data_dict["pixel_values"].to(torch.bfloat16)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 1.0)
    top_p = kwargs.get('top_p', 0.9 if do_sample else 1.0)
    top_k = kwargs.get('top_k', 20 if do_sample else 50)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    with torch.inference_mode():
        output_ids = model.generate(
            **data_dict,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def INFERENCES(model_path):
    config = AutoConfig.from_pretrained(model_path)

    # judge model type
    model_type = config.model_type
    if 'videollama2' in model_type.lower():
        return videollama2.model_init, videollama2.mm_infer
    elif 'videollama3' in model_type.lower():
        # NOTE: remote version of VideoLLaMA3
        if config.vision_encoder is None:
            return model_init, mm_infer
        # NOTE: local version of VideoLLaMA3
        return videollama3.model_init, videollama3.mm_infer
    else:
        raise NotImplementedError(f"Model path {model_path} not recognized.")
