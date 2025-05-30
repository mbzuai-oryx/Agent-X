import os
import json
from multiagent_evaluation import *
import json
import argparse
parser = argparse.ArgumentParser()



if __name__ == "__main__":

    # Adding arguments
    parser.add_argument("--save_path",default="eval_results.json", help = "path to Output evaluation results json file")
    parser.add_argument("--pred_path", default="qwenvl_results_final.json", help = "path to model predictions json file")
    parser.add_argument("--gt_data_path", default="./AgentX/data.json", help = "path to ground truth json file")

    # Read arguments from command line
    args = parser.parse_args()

        ## read the gt reason data
    with open(args.gt_data_path, "r") as g:
        gt_data = json.load(g)
    g.close()

    with open(args.pred_path, "r") as f:
        pred_data = json.load(f)
    f.close()

    save_dict = {}

    for data in pred_data:
        key = list(data.keys())[0]
        print(key)
        scores = {
                    "grounding_accuracy": None,
                    "precision_score": None,
                    "tool_accuray": None,
                    "faithfulness_accuray": None,
                    "goal_accuray": None,
                    "toolset_accuray": None,
                    "step_score": None,
                    "context_score": None,
                    "clarity_penalty": None,
                    "factual_precision": None,
                    "semantic_accuracy": None,
                    "reward_score": None
                }
        try:

            for key, value in data.items():
                
                gt_query = gt_data[key][0]["query"]
                gt_tools = gt_data[key][0]["tool_metadata"].keys()
                gt_tool_metadata = gt_data[key][0]["tool_metadata"]
                gt_reasoning_steps = gt_data[key][0]["reasoning_steps"]
                gt_final_answer = gt_data[key][0]["final_answer"] 

                if 'reasoning_steps' in value.keys():
                    pred_reasoning_steps = value['reasoning_steps']
                else:
                    pred_tools = '[]'
                    pred_reasoning_steps = '[]'
                if 'final_answer' in value.keys():
                    pred_final_answer = value['final_answer']
                else:
                    pred_final_answer = '[]'

                gt_final = {"query": gt_query, "GT reasoning steps":gt_reasoning_steps, "GT final answer": gt_final_answer}
                gt_query_type = gt_data[key][0]["query_type"]

                grounding_accuracy = get_grounding_score(str(gt_final), str(pred_reasoning_steps))
                precision_score = get_precision_score(str(gt_final), str(pred_reasoning_steps))
                tool_accuray = get_tool_accuray(str(gt_tool_metadata), str(pred_reasoning_steps))
                faithfulness_accuray = get_faithfulness_accuray(str(gt_final), str(pred_reasoning_steps))
                goal_accuray = get_goal_accuray(str(gt_final_answer), gt_query_type, str(pred_final_answer))
                toolset_accuray = get_toolset_accuray(str(gt_final), str(pred_reasoning_steps))
                step_score = get_step_score(str(pred_reasoning_steps))
                context_score = get_context_score(str(gt_final), str(pred_reasoning_steps))
                clarity_penalty = get_clarity_penalty(str(pred_reasoning_steps))
                factual_precision = get_factual_precision(str(gt_final), str(pred_reasoning_steps))
                semantic_accuracy = get_semantic_accuracy(str(gt_final), str(pred_reasoning_steps), str(pred_final_answer))
                reward_score = get_reward_score(str(gt_query), str(pred_reasoning_steps))

            scores = {
                    "grounding_accuracy": grounding_accuracy,
                    "precision_score": precision_score,
                    "tool_accuray": tool_accuray,
                    "faithfulness_accuray": faithfulness_accuray,
                    "goal_accuray": goal_accuray,
                    "toolset_accuray": toolset_accuray,
                    "step_score": step_score,
                    "context_score": context_score,
                    "clarity_penalty": clarity_penalty,
                    "factual_precision": factual_precision,
                    "semantic_accuracy": semantic_accuracy,
                    "reward_score": reward_score
                }
            print(scores)
            save_dict[key] = scores
            with open(args.save_path, "w") as f:
                json.dump(save_dict, f, indent=2)
        except Exception as e:
            print(e)
            save_dict[key] = scores
