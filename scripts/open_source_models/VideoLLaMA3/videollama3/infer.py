import sys
sys.path.append('./')
from videollama3 import disable_torch_init, model_init, mm_infer
from videollama3.mm_utils import load_video, load_images


def main():
    disable_torch_init()

    modal = "text"
    conversation = [
        {
            "role": "user",
            "content": "What is the color of bananas?",
        }
    ]

    modal = "image"
    frames = load_images("assets/sora.png")[0]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is the woman wearing?"},
            ]
        }
    ]

    modal = "video"
    frames, timestamps = load_video("assets/cat_and_chicken.mp4", fps=1, max_frames=180)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": "What is the cat doing?"},
            ]
        }
    ]

    model_path = "/path/to/your/model"
    model, processor = model_init(model_path)

    inputs = processor(
        images=[frames] if modal != "text" else None,
        text=conversation,
        merge_size=2 if modal == "video" else 1,
        return_tensors="pt",
    )

    output = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal
    )
    print(output)


if __name__ == "__main__":
    main()
