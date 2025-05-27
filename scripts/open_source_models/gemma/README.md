## Run Gemma 

### Create conda environment

```
conda create -n gemma python==3.10
pip install -r requirements.txt
conda activate gemma
```

Access the gemma weights from [here](https://huggingface.co/google/gemma-3-4b-it).


### Run inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

