## Run InternVL

### Create conda environment

```
conda create -n internvl python=3.9
conda activate internvl
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation

```

### Model weights
Access the InternVL-2.5-8B and InternVL3-8B weights below:

<table>
  <tr>
    <th>Model Name</th>
    <th>HF&nbsp;Link</th>
  </tr>
  <tr>
    <td>InternVL2_5-8B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-8B">🤗 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-8B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-8B">🤗 link</a></td>
  </tr>

</table>


### Inference 
Depending on the version of InternVL, change the name of the model accordingly inside the run_inference.py file.

```
python run_inference.py --save_path <path to json for saving inference results> \
					--base_path <path to base data files folder> \
				 	--tool_data_path <path to tool metadata json file> \
 					--gt_data_path <path to groundtruth data json>
```

