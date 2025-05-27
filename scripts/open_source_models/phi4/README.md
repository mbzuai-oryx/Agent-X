## Run Phi4-Multimodal-Instruct 

### Create conda environment

```
conda create -n phi4 python==3.10
pip install -r requirements.txt
```

Access the phi4-multimodal-instruct weights from [here](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).


### Inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

