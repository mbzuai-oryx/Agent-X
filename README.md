
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
<!-- centred logo -->
<div align="center" style="margin:24px 0;">
  <img src="logo.png" width="85%" />
</div>

<!-- bottom full-width GIF -->
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>
<h1 align="left" style="margin:24px 0;">
  Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks
</h1>


<div align="left">

[![](https://img.shields.io/badge/website-Agent-X-purple?style=plastic&logo=Google%20chrome)](https://Agent-X.stanford.edu/)
[![](https://img.shields.io/badge/Dataset-online-yellow?style=plastic&logo=Hugging%20face)](https://huggingface.co/datasets/snap-stanford/Agent-X)
[![](https://img.shields.io/badge/SKB_Explorer-online-yellow?style=plastic&logo=Hugging%20face)](https://Agent-X.stanford.edu/skb_explorer.html)
[![](https://img.shields.io/badge/Huggingface_Learderboard-online-yellow?style=plastic&logo=Hugging%20face)](https://huggingface.co/spaces/snap-stanford/Agent-X-leaderboard)
[![](https://img.shields.io/badge/Arxiv-paper-red?style=plastic&logo=arxiv)](https://arxiv.org/abs/2404.13207)
[![](https://img.shields.io/badge/pip-Agent-X--qa-brightgreen?style=plastic&logo=Python)](https://pypi.org/project/Agent-X-qa/) 
[![](https://img.shields.io/badge/doc-online-blue?style=plastic&logo=Read%20the%20Docs)](https://Agent-X.stanford.edu/docs/index.html)
[![](https://img.shields.io/badge/-Linkedin-blue?style=plastic&logo=Linkedin)](https://www.linkedin.com/posts/leskovec_reduce-llm-hallucinations-with-rag-over-textual-activity-7190745116339302401-da4n?utm_source=share&utm_medium=member_desktop) 
[![](https://img.shields.io/badge/-Twitter-cyan?style=plastic&logo=X)](https://twitter.com/ShirleyYXWu/status/1784970920383402433) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


#### Authors: [Tajamul Ashraf](https://www.tajamulashraf.com)\*, [Amal Saqib](https://github.com/amalsaqib)\*, [Hanan Ghani](https://github.com/hananshafi), [Muhra AlMahri](https://github.com/muhraalmahri), [Yuhao Li](https://ae.linkedin.com/in/yuhao-li-610ba7288), [Noor Ahsan](https://www.linkedin.com/in/noorahsan1415/), [Umair Nawaz](https://scholar.google.com/citations?user=w7N4wSYAAAAJ&hl=en), [Jean Lahoud](https://scholar.google.com/citations?user=LsivLPoAAAAJ&hl=en), [Hisham Cholakkal](https://mbzuai.ac.ae/study/faculty/hisham-cholakkal/), [Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/), [Philip H.S. Torr](https://en.wikipedia.org/wiki/Philip_Torr), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home), [Rao Muhammad Anwer](https://mbzuai.ac.ae/study/faculty/rao-muhammad-anwer/), and [Salman Khan](https://salman-h-khan.github.io/)


\* Equally contribution, **Correspondence:** [Tajamul Ashraf](https://www.tajamulashraf.com), [Amal Saqib](https://github.com/amalsaqib).
<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" />
</div>



## 🚀 Updates

 [2024-07-03]: **Released evaluation & deployment code for Agent-X** 🔥🔥🔥
 
[2024-07-01]:  **Published the Agent-X dataset on Hugging Face** 🎉🎉🎉

## 🌟 Introduction

>Current tool-use tests for vision-centric LLMs rely on single-turn, synthetic queries and text-only inputs, so they miss the real-world challenge of multi-step, multimodal reasoning. Agent-X closes this gap with 828 authentic tasks spanning images, videos, and mixed-modal instructions across six domains—from web browsing to autonomous driving. Each task demands explicit, step-by-step decisions and judicious tool use, and our evaluation scores every reasoning step as well as the overall chain. Even top models (GPT, Gemini, Qwen) solve fewer than half of these tasks, exposing major bottlenecks and pointing the way for future research.


## 🌟 What is Agent-X?

Agent-X is a benchmark for assessing deep-reasoning and tool-use skills of vision-centric LLM agents in real-world settings. It highlights three key aspects:
- **Authentic multi-step tasks.** The benchmark offers 828 human-authored tasks with implicit tool use and sequential planning requirements, spanning six domains such as web browsing, surveillance, autonomous driving, sports, and math reasoning. 
- **Real deployed tools.** Agent-X supplies an evaluation platform stocked with perception, web, manipulation, math, and data-processing tools, compelling agents to choose and apply the right tool at each reasoning step.
- **Diverse multimodal contexts.**  Every task is paired with real images, multi-image comparisons, or video clips—plus textual instructions, closely mirroring the visual complexity of real-world scenarios.
<div align="center">
 <img src="data_teasor.png" width="800"/>
</div>

The comparison of Agent-Xqueries with AI-generated queries is shown in the table below. The steps and tool types for queries in ToolBench and m\&m's are explicitly stated, as marked in red and blue. The queries in APIBench are simple, only containing one step. Our GTA's queries are both step-implicit and tool-implicit.
<div align="center">
 <img src="table_comparison.png" width="800"/>
</div>


## 📚 Dataset Statistics
Overview of the Agent-X benchmark. Key data statistics,  overall frequency of the tool
usage, number of steps, and distribution of tasks across the six vision-centric environments.


<div align="center">
 <img src="data_statistics.png" width="800"/>
</div>

The detailed information of 14 tools are shown in the table below.

<div align="center">
 <img src="pipeline.png" width="800"/>
</div>

## 🏆 Leaderboard, July 2024

We evaluate the language models in two modes:
- **Step-by-step mode.** It is designed to evaluate the model's fine-grained tool-use capabilities. In this mode, the model is provided with the initial $n$ steps of the reference tool chain as prompts, with the expectation to predict the action in step $n+1$. Four metrics are devised under step-by-step mode: ***InstAcc*** (instruction following accuracy), ***ToolAcc*** (tool selection accuracy), ***ArgAcc*** (argument prediction accuracy), and ***SummAcc*** (answer summarizing accuracy).

- **End-to-end mode.** It is designed to reflect the tool agent's actual task executing performance. In this mode, the model actually calls the tools and solves the problem by itself. We use ***AnsAcc*** (final answer accuracy) to measure the accuracy of the execution result. Besides, we calculate four ***F1 scores of tool selection: P, L, O, C*** in perception, operation, logic, and creativity categories, to measure the tool selection capability. 

Here is the performance of various LLMs on GTA. Inst, Tool, Arg, Summ, and Ans denote InstAcc, ToolAcc, ArgAcc SummAcc, and AnsAcc, respectively. P, O, L, C denote the F1 score of tool selection in Perception, Operation, Logic, and Creativity categories. ***Bold*** denotes the best score among all models. <ins>*Underline*</ins> denotes the best score under the same model scale. ***AnsAcc*** reflects the overall performance.

### With GPT-4o as a judge
| **Model** | Ground<sub>s</sub> | Tool<sub>p</sub> | Tool<sub>acc</sub> | Fact<sub>acc</sub> | Context<sub>s</sub> | Fact<sub>p</sub> | Sem<sub>acc</sub> | Ground<sub>acc</sub> | Ground<sub>acc</sub><sup>*</sup> | Tool<sub>acc</sub><sup>s</sup> |
|----------|--------------------|------------------|--------------------|--------------------|---------------------|------------------|-------------------|-----------------------|------------------------------|---------------------------|
| *Open-source* |||||||||||
| Phi-4-VL-Instruct|0.13|0.21|0.24|0.61|0.19|0.47|0.40|0.11|0.26|0.42|
| InternVL-2.5-8B|0.45|0.31|0.47|0.68|*0.47*|0.52|0.60|*0.28*|*0.55*|0.58|
| Gemma-3-4B|0.26|0.30|<ins><strong>0.78</strong></ins>|0.61|*0.54*|0.38|0.54|0.27|<ins><strong>0.67</strong></ins>|0.60|
| InternVL-3-8B|*0.46*|*0.34*|0.54|0.68|0.45|<ins><strong>0.70</strong></ins>|0.40|0.20|0.59|*0.62*|
| VideoLLaMA-3-7B|0.45|0.28|0.46|0.65|0.46|*0.62*|0.54|0.28|0.54|0.54|
| Qwen-2.5-VL-7B|<ins><strong>0.54</strong></ins>|<ins><strong>0.43</strong></ins>|*0.63*|<ins><strong>0.75</strong></ins>|<ins><strong>0.57</strong></ins>|0.56|<ins><strong>0.67</strong></ins>|<ins><strong>0.36</ins></strong>|*0.65*|<ins><strong>0.67</strong></ins>|
| Pixtral-12B|0.12|0.20|*0.63*|0.45|0.19|0.26|0.34|0.07|<ins><strong>0.55</strong></ins>|0.54|
| LLaMA-3.2-11B-Vision|0.03|0.15|0.14|*0.70*|0.08|<ins><strong>0.70</strong></ins>|0.24|0.07|0.26|0.42|
| Kimi-VL-A3B-Thinking|0.26|0.19|0.5*|0.62|0.42|0.52|*0.65*|*0.29*|0.29|0.48|
| mPLUG-Owl3-7B-240728|0.10|0.14|0.30|0.49|0.25|0.32|0.37|*0.11*|0.26|0.50|
| *Closed-source* |  |  |  |  |  |  |  |  |  |  |
| Gemini-1.5-Pro  | 0.43 | 0.23 | *0.84* | 0.62 | 0.45 | 0.53 | 0.62 | 0.04 | 0.56 | 0.48 |
| Gemini-2.5-Pro  | 0.40 | *0.36* | 0.81 | *0.72* | 0.48 | 0.64 | *0.73* | *0.40* | 0.56 | *0.62* |
| GPT-4o          | <ins><strong>0.60</strong></ins> | <ins><strong>0.47</strong></ins> | 0.72 | <ins><strong>0.81</strong></ins> | <ins><strong>0.57</strong></ins> | <ins><strong>0.79</strong></ins> | 0.59 | 0.37 | <ins><strong>0.70</strong></ins> | <ins><strong>0.68</strong></ins> |
| OpenAI o4-mini  | *0.42* | 0.32 | <ins><strong>0.89</strong></ins> | 0.71 | *0.51* | *0.60* | <ins><strong>0.80</strong></ins> | <ins><strong>0.45</strong></ins> | *0.67* | 0.63 |

_Best values are **bold + underlined**; second-best values are *italic*._

### With Qwen-15B  as a judge
| **Model** | Ground<sub>s</sub> | Tool<sub>p</sub> | Tool<sub>acc</sub> | Fact<sub>acc</sub> | Context<sub>s</sub> | Fact<sub>p</sub> | Sem<sub>acc</sub> | Ground<sub>acc</sub> | Ground<sub>acc</sub><sup>*</sup> | Tool<sub>acc</sub><sup>s</sup> |
|----------|--------------------|------------------|--------------------|--------------------|---------------------|------------------|-------------------|-----------------------|------------------------------|---------------------------|
| *Open-source* |||||||||||
| Phi-4-VL-Instruct [1]        | 0.27 | 0.11 | 0.32 | 0.54 | 0.39 | 0.59 | 0.46 | 0.16 | 0.35 | 0.39 |
| InternVL2.5-8B [12]          | 0.38 | 0.16 | 0.49 | 0.63 | 0.51 | 0.61 | 0.55 | 0.29 | 0.53 | 0.53 |
| Gemma-3-4B [57]              | *0.50* | 0.24 | *0.67* | 0.74 | *0.66* | 0.59 | *0.74* | 0.30 | **_0.68_** | *0.68* |
| InternVL3-8B [81]            | 0.41 | 0.16 | 0.51 | 0.71 | 0.61 | 0.60 | 0.69 | 0.23 | 0.51 | 0.62 |
| VideoLLaMA3-7B [75]          | 0.39 | 0.15 | 0.40 | 0.68 | 0.56 | 0.60 | 0.68 | 0.27 | 0.53 | 0.56 |
| Qwen2.5-VL-7B [8]            | **_0.51_** | **_0.27_** | 0.63 | *0.77* | 0.66 | 0.64 | **_0.77_** | *0.37* | 0.62 | 0.67 |
| Pixtral-12B [3]              | 0.30 | 0.17 | 0.61 | 0.59 | 0.50 | 0.42 | 0.58 | 0.10 | *0.68* | 0.58 |
| LLaMA-3.2-11B-Vision [17]    | 0.16 | 0.06 | 0.12 | 0.49 | 0.17 | **_0.74_** | 0.20 | 0.10 | 0.11 | 0.15 |
| Kimi-VL-A3B-Thinking [58]    | *0.47* | *0.20* | 0.59 | **_0.79_** | 0.64 | *0.68* | *0.74* | **_0.35_** | 0.60 | 0.62 |
| mPLUG-Owl3-7B-240728 [73]    | 0.30 | 0.11 | 0.31 | 0.59 | 0.48 | 0.48 | 0.56 | 0.16 | 0.45 | 0.48 |
| *Closed-source* |||||||||||
| Gemini-1.5-Pro [56]          | 0.57 | 0.36 | *0.80* | *0.82* | 0.73 | *0.76* | 0.63 | 0.05 | **_0.77_** | 0.71 |
| Gemini-2.5-Pro [55]          | *0.63* | *0.40* | **_0.84_** | **_0.86_** | *0.76* | **_0.80_** | *0.83* | *0.50* | 0.74 | *0.72* |
| GPT-4o [21]                  | 0.46 | 0.27 | 0.63 | 0.72 | 0.59 | 0.75 | 0.69 | 0.44 | 0.48 | 0.56 |
| OpenAI-o4-mini [45]          | **_0.63_** | 0.35 | *0.86* | *0.89* | **_0.78_** | 0.79 | **_0.88_** | **_0.53_** | *0.64* | **_0.69_** |



_Best values are **bold + underlined**; second-best values are *italic*._


## 🚀 Evaluate on GTA

### Prepare Agent-XDataset
1. Clone this repo.
```shell
git clone https://github.com/open-compass/GTA.git
cd GTA
```
2. Download the dataset from [release file](https://github.com/open-compass/GTA/releases/download/v0.1.0/gta_dataset.zip).
```shell
mkdir ./opencompass/data
```
Put it under the folder ```./opencompass/data/```. The structure of files should be:
```
GTA/
├── agentlego
├── opencompass
│   ├── data
│   │   ├── gta_dataset
│   ├── ...
├── ...
```

### Prepare Your Model
1. Download the model weights.
```shell
pip install -U huggingface_hub
# huggingface-cli download --resume-download hugging/face/repo/name --local-dir your/local/path --local-dir-use-symlinks False
huggingface-cli download --resume-download Qwen/Qwen1.5-7B-Chat --local-dir ~/models/qwen1.5-7b-chat --local-dir-use-symlinks False
```
2. Install [LMDeploy](https://github.com/InternLM/lmdeploy).
```shell
conda create -n lmdeploy python=3.10
conda activate lmdeploy
```
For CUDA 12:
```shell
pip install lmdeploy
```
For CUDA 11+:
```shell
export LMDEPLOY_VERSION=0.4.0
export PYTHON_VERSION=310
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```
3. Launch a model service.
```shell
# lmdeploy serve api_server path/to/your/model --server-port [port_number] --model-name [your_model_name]
lmdeploy serve api_server ~/models/qwen1.5-7b-chat --server-port 12580 --model-name qwen1.5-7b-chat
```
### Deploy Tools
1. Install [AgentLego](https://github.com/InternLM/agentlego).
```shell
conda create -n agentlego python=3.11.9
conda activate agentlego
cd agentlego
pip install -r requirements_all.txt
pip install agentlego
pip install -e .
mim install mmengine
mim install mmcv==2.1.0
```
Open ```~/anaconda3/envs/agentlego/lib/python3.11/site-packages/transformers/modeling_utils.py```, then set ```_supports_sdpa = False``` to ```_supports_sdpa = True``` in line 1279.

2. Deploy tools for Agent-Xbenchmark.

To use the GoogleSearch and MathOCR tools, you should first get the Serper API key from https://serper.dev, and the Mathpix API key from https://mathpix.com/. Then export these keys as environment variables.

```shell
export SERPER_API_KEY='your_serper_key_for_google_search_tool'
export MATHPIX_APP_ID='your_mathpix_key_for_mathocr_tool'
export MATHPIX_APP_KEY='your_mathpix_key_for_mathocr_tool'
```

Start the tool server.

```shell
agentlego-server start --port 16181 --extra ./benchmark.py  `cat benchmark_toollist.txt` --host 0.0.0.0
```
### Start Evaluation
1. Install [OpenCompass](https://github.com/open-compass/opencompass).
```shell
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
cd agentlego
pip install -e .
cd ../opencompass
pip install -e .
```
huggingface_hub==0.25.2 (<0.26.0)
transformers==4.40.1
2. Modify the config file at ```configs/eval_gta_bench.py``` as below.

The ip and port number of **openai_api_base** is the ip of your model service and the port number you specified when using lmdeploy.

The ip and port number of **tool_server** is the ip of your tool service and the port number you specified when using agentlego.

```python
models = [
  dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        tool_server='http://10.140.0.138:16181',
        tool_meta='data/gta_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```

If you infer and evaluate in **step-by-step** mode, you should comment out **tool_server** and enable **tool_meta** in ```configs/eval_gta_bench.py```, and set infer mode and eval mode to **every_with_gt** in ```configs/datasets/gta_bench.py```:
```python
models = [
  dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        # tool_server='http://10.140.0.138:16181',
        tool_meta='data/gta_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```
```python
gta_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt'),
)
gta_bench_eval_cfg = dict(evaluator=dict(type=GTABenchEvaluator, mode='every_with_gt'))
```

If you infer and evaluate in **end-to-end** mode, you should comment out **tool_meta** and enable **tool_server** in ```configs/eval_gta_bench.py```, and set infer mode and eval mode to **every** in ```configs/datasets/gta_bench.py```:
```python
models = [
  dict(
        abbr='qwen1.5-7b-chat',
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=10,
        llm=dict(
            type=OpenAI,
            path='qwen1.5-7b-chat',
            key='EMPTY',
            openai_api_base='http://10.140.1.17:12580/v1/chat/completions',
            query_per_second=1,
            max_seq_len=4096,
            stop='<|im_end|>',
        ),
        tool_server='http://10.140.0.138:16181',
        # tool_meta='data/gta_dataset/toolmeta.json',
        batch_size=8,
    ),
]
```
```python
gta_bench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="""{questions}""",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode='every'),
)
gta_bench_eval_cfg = dict(evaluator=dict(type=GTABenchEvaluator, mode='every'))
```

3. Infer and evaluate with OpenCompass.
```shell
# infer only
python run.py configs/eval_gta_bench.py --max-num-workers 32 --debug --mode infer
```
```shell
# evaluate only
# srun -p llmit -q auto python run.py configs/eval_gta_bench.py --max-num-workers 32 --debug --reuse [time_stamp_of_prediction_file] --mode eval
srun -p llmit -q auto python run.py configs/eval_gta_bench.py --max-num-workers 32 --debug --reuse 20240628_115514 --mode eval
```
```shell
# infer and evaluate
python run.py configs/eval_gta_bench.py -p llmit -q auto --max-num-workers 32 --debug
```


# 📝 Citation
If you use Agent-Xin your research, please cite the following paper:
```

}
```

