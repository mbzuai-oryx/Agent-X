import os
import os.path as osp

import gradio as gr

HEADER = """
"""


class VideoLLaMA3GradioInterface(object):

    def __init__(self, model_client, example_dir=None, **server_kwargs):
        self.model_client = model_client
        self.server_kwargs = server_kwargs

        self.image_formats = ("png", "jpg", "jpeg")
        self.video_formats = ("mp4",)

        image_examples, video_examples = [], []
        if example_dir is not None:
            example_files = [
                osp.join(example_dir, f) for f in os.listdir(example_dir)
            ]
            for example_file in example_files:
                if example_file.endswith(self.image_formats):
                    image_examples.append([example_file])
                elif example_file.endswith(self.video_formats):
                    video_examples.append([example_file])

        with gr.Blocks() as self.interface:
            gr.Markdown(HEADER)
            with gr.Row():
                chatbot = gr.Chatbot(type="messages", elem_id="chatbot", height=710)

                with gr.Column():
                    with gr.Tab(label="Input"):

                        with gr.Row():
                            input_video = gr.Video(sources=["upload"], label="Upload Video")
                            input_image = gr.Image(sources=["upload"], type="filepath", label="Upload Image")

                        if len(image_examples):
                            gr.Examples(image_examples, inputs=[input_image], label="Example Images")
                        if len(video_examples):
                            gr.Examples(video_examples, inputs=[input_video], label="Example Videos")

                        input_text = gr.Textbox(label="Input Text", placeholder="Type your message here and press enter to submit")

                        submit_button = gr.Button("Generate")

                    with gr.Tab(label="Configure"):
                        with gr.Accordion("Generation Config", open=True):
                            do_sample = gr.Checkbox(value=True, label="Do Sample")
                            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, label="Temperature")
                            top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, label="Top P")
                            max_new_tokens = gr.Slider(minimum=0, maximum=4096, value=2048, step=1, label="Max New Tokens")

                        with gr.Accordion("Video Config", open=True):
                            fps = gr.Slider(minimum=0.0, maximum=10.0, value=1, label="FPS")
                            max_frames = gr.Slider(minimum=0, maximum=256, value=180, step=1, label="Max Frames")

            input_video.change(self._on_video_upload, [chatbot, input_video], [chatbot, input_video])
            input_image.change(self._on_image_upload, [chatbot, input_image], [chatbot, input_image])
            input_text.submit(self._on_text_submit, [chatbot, input_text], [chatbot, input_text])
            submit_button.click(
                self._predict,
                [
                    chatbot, input_text, do_sample, temperature, top_p, max_new_tokens,
                    fps, max_frames
                ],
                [chatbot],
            )

    def _on_video_upload(self, messages, video):
        if video is not None:
            # messages.append({"role": "user", "content": gr.Video(video)})
            messages.append({"role": "user", "content": {"path": video}})
        return messages, None

    def _on_image_upload(self, messages, image):
        if image is not None:
            # messages.append({"role": "user", "content": gr.Image(image)})
            messages.append({"role": "user", "content": {"path": image}})
        return messages, None

    def _on_text_submit(self, messages, text):
        messages.append({"role": "user", "content": text})
        return messages, ""

    def _predict(self, messages, input_text, do_sample, temperature, top_p, max_new_tokens,
                 fps, max_frames):
        if len(input_text) > 0:
            messages.append({"role": "user", "content": input_text})
        new_messages = []
        contents = []
        for message in messages:
            if message["role"] == "assistant":
                if len(contents):
                    new_messages.append({"role": "user", "content": contents})
                    contents = []
                new_messages.append(message)
            elif message["role"] == "user":
                if isinstance(message["content"], str):
                    contents.append(message["content"])
                else:
                    media_path = message["content"][0]
                    if media_path.endswith(self.video_formats):
                        contents.append({"type": "video", "video": {"video_path": media_path, "fps": fps, "max_frames": max_frames}})
                    elif media_path.endswith(self.image_formats):
                        contents.append({"type": "image", "image": {"image_path": media_path}})
                    else:
                        raise ValueError(f"Unsupported media type: {media_path}")

        if len(contents):
            new_messages.append({"role": "user", "content": contents})

        if len(new_messages) == 0 or new_messages[-1]["role"] != "user":
            return messages

        generation_config = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens
        }

        streamer = self.model_client.submit({"conversation": new_messages, "generation_config": generation_config})
        messages.append({"role": "assistant", "content": ""})
        for token in streamer:
            messages[-1]['content'] += token
            yield messages

    def launch(self):
        self.interface.launch(**self.server_kwargs)
