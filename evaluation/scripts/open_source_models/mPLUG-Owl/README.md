## Run mPLUG-Owl3

### Create conda environment

```
conda create -n mplugowl python==3.10
conda activate mplugowl 
pip install -r requirements.txt
```

Access the mPLUG-Owl3-7B from [here](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728).


### Inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

