## Run GPT

### Install necessary packages
- Python>=3.10
- openai==0.28.0

use a valid Openai key to run the inference. You can use either gpt-4o or o4-mini to run the inference.

### Run inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json> \
					--gpt_type gpt-4o
```



## Run Gemini

### Install necessary packages
- Python>=3.10
- pip install google-generativeai

Use a valid gemini key to run the inference. You can use either gemini-1.5-pro or gemini-2.5-pro to run the inference.

```
export GOOGLE_API_KEY=""
```

### Run inference

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \pip install google-generativeai
 					--gt_data_path <path to groundtruth data json> \
					--gemini_type gemini-2.5-pro
```


