## Run Pixtral

### Create conda environment

```
conda create -n pixtral python==3.10
conda activate pixtral
pip install -r requirements.txt
```

Access the pixtral-12b weights from [here](https://huggingface.co/mistral-community/pixtral-12b).


### Inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

