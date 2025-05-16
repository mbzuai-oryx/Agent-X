import torch
from transformers import AutoModelForCausalLM, AutoProcessor


# NOTE: transformers==4.46.3 is recommended for this script
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@torch.inference_mode()
def infer(conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response


# Video conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": "./assets/cat_and_chicken.mp4", "fps": 1, "max_frames": 180}},
            {"type": "text", "text": "What is the cat doing? Please describe the scene, the obejcts and the actions in detail."},
        ]
    },
]
print(infer(conversation))

# Image conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": {"image_path": "./assets/sora.png"}},
            {"type": "text", "text": "Please describe the model?"},
        ]
    }
]
print(infer(conversation))

# Mixed conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": "./assets/cat_and_chicken.mp4", "fps": 1, "max_frames": 180}},
            {"type": "text", "text": "What is the relationship between the video and the following image?"},
            {"type": "image", "image": {"image_path": "./assets/sora.png"}},
        ]
    }
]
print(infer(conversation))

# Plain text conversation
conversation = [
    {
        "role": "user",
        "content": "What is the color of bananas?",
    }
]
print(infer(conversation))
