import json
import os

tool_metadata_path = "/share/data/drive_1/hanan/multiagent_eval_data/toolmeta.json"
with open(tool_metadata_path, "r") as f:
    meta_data = json.load(f)

meta_data = json.dumps(meta_data)

instruction_prompt = (
        "You are a multi-modal intelligent assistant. You are provided with:"
        "- A text query,"
        "- An image or video,"
        "- A set of tools to assist with your reasoning with meta data of tools given as follows:\n"

        f"{meta_data}\n\n"

        "Your objective is to answer the query based on the given visual content "
        "by choosing and using the most appropriate tools. You must reason step-by-step. "
        "Each reasoning step should include: \n\n"

    "reasoning_step_format:"
        "{task: Describe the sub-task being performed.,"
        "{tool_used: Specify the tool selected and justify its choice.,"
        "{tool_output: Provide the tool's output.,"
        "{thought: Explain the significance of the output and how it contributes to answering the query.\n\n"

    "constraints:"
        "Use only necessary tools and justify their usage.,"
        "Ensure each step is self-contained and clearly explained.,"
        "Maintain transparency in decision-making and reasoning.\n\n"

    "final_output\n"
        "final_answer: Provide a clear and concise answer based on all previous steps.\n\n"

    "example:\n" 
        "query: What is the man doing in the video?"
        "steps:"
            "{"
                "task: Identify key frames containing activity.,"
                "tool_used: Scene Segmentation Tool,"
                "tool_output: Frames 40-80 show continuous motion.,"
                "thought: These frames likely contain the main action.}"

            "{"
  
                "task: Recognize the action happening in the selected frames.,",
                "tool_used: Action Recognition Model,"
                "tool_output: Man is playing guitar,"
                "thought: The model confirms that the man is engaged in a musical activity.}\n\n"

        "final_answer: The man is playing a guitar in the video."

)
