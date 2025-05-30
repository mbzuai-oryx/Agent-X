## GPT-4o as a judge

### Install necessary packages
- Python>=3.10
- openai==0.28.0

use a valid Openai key to run the inference.

```
export OPENAI_API_KEY=""
```

### Run evaluation

```
python run_eval_gpt_as_judge.py --save_path <path to json for saving evaluation results> \
 				--gt_data_path <path to groundtruth data json file> \
				--pred_path <path to model predictions json file>
```



## Qwen3-14B as a Judge

### Install necessary packages
```
conda create -n qwen python=3.10
conda activate qwen
pip install -r requirements.txt
pip install transformers==4.51.3 accelerate
```

### Model weights
Access the Qwen3-14B weights below:

<table>
  <tr>
    <th>Model Name</th>
    <th>HF&nbsp;Link</th>
  </tr>
  <tr>
    <td>Qwen3-14B-Instruct</td>
    <td><a href="https://huggingface.co/Qwen/Qwen3-14B">🤗 link</a></td>
  </tr>
</table>

### Run evaluation

```
python run_eval_qwen_as_judge.py --save_path <path to json for saving evaluation results> \
 				 --gt_data_path <path to groundtruth data json file> \
				 --pred_path <path to model predictions json file>
```


