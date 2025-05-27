## Run Kimi-VL

### Create conda environment

```
conda create -n kimivl python==3.10
pip install -r requirements.txt
conda activate kimivl
```

Access the Kimi-VL-A3B-Thinking weights from [here](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking).


### Inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

