## Run VideoLLaMA3

### 🛠️ Requirements and Installation

Basic Dependencies:

* Python >= 3.10
* Pytorch >= 2.4.0
* CUDA Version >= 11.8
* transformers >= 4.46.3

Make conda environment and install required packages:

```bash
conda create -n videollama3 python==3.10
pip install torch==2.4.0 torchvision==0.17.0 --extra-index-url https://download.pytorch.org/whl/cu118

pip install flash-attn --no-build-isolation
pip install transformers==4.46.3 accelerate==1.0.1
pip install decord ffmpeg-python imageio opencv-python
```

Access the VideoLLaMA3 weights below.

| Model                | Base Model   | HF Link                                                      |
| -------------------- | ------------ | ------------------------------------------------------------ |
| VideoLLaMA3-7B-Image | Qwen2.5-7B   | [DAMO-NLP-SG/VideoLLaMA3-7B-Image](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B-Image) |


### Inference

```
python run_inference.py --save_path <path to json for saving inference results> I am running a few minutes late; my previous meeting is running over.
					--base_path <path to base data files folder> I am running a few minutes late; my previous meeting is running over.
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
