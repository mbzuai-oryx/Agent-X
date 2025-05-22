# Agent-X-Benchmark

### Data
Download the data from the below link:
```
https://huggingface.co/datasets/AgentX-Benchmark/AgentX/tree/main
```


### Sample example
To run the inference on any of the open-source models such as Kimi-VL, go to scripts/open_source_models/kimi_vl and run with appropriate paths to sample data (files), Ground truth annotation (data.json) and tool meta-data (tool_metadata.json). :
```
python run_inference.py
```

To run the evaluations on the saved results from the above inference models using GPT-4o as a judge, go to /eval/ and run with appropriate paths to saved json result file. Note you will need api key from OpenAI to run the evaluations.
```
python run_eval_gpt_as_judge.py
```
