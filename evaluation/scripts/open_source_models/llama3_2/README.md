## Run Llama-3.2

### Create conda environment

```
conda create -n llama python==3.10
conda activate llama
pip install -r requirements.txt
```

Access the Llama-3.2-11B-Vision-Instruct weights from [here](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).


### Inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

