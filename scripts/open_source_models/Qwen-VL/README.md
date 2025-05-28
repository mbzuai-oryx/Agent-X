## Run Qwen-VL

### Create conda environment

```
conda create -n qwenvl python=3.10
conda activate qwenvl
pip install -r requirements.txt
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
```

### Model weights
Access the Qwen2.5-VL-8B-Instruct weights below:

<table>
  <tr>
    <th>Model Name</th>
    <th>HF&nbsp;Link</th>
  </tr>
  <tr>
    <td>Qwen2.5-VL-8B-Instruct</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">🤗 link</a></td>
  </tr>
</table>


### Inference 

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

