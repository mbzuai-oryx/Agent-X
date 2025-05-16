import openai

openai.api_key = "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"

###################### GROUNDING SCORE #####################################

def get_grounding_score(target,pred):

    evaluation_prompt = """You are an evaluation assistant measuring the groundedness of each reasoning step performed by a vision-language agent.

        You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.
        You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought.

        Your task is to assess whether each agent step is visually and contextually grounded — that is, whether the task, the selected tool, the tool output, and the agent's thought process are all supported by the GT.

        Evaluation Guidelines:
        If the number of steps in the GT and the model differ:
        - Any extra model step beyond the GT steps should receive a score of 0 (hallucinated or unjustified).
        - Any GT step that the model omits should also receive a score of 0 (missing reasoning).
        The Grounding Score for the full query is computed as the average of the per-step scores.

        Assess the following per step:
        - Is the task relevant to the query and aligned with the GT?
        - Is the tool appropriate for this task?
        - Is the output consistent with what's in the GT?
        - Is the thought grounded and logically aligned?

        Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents all aspects are grounded (task, tool, output, and thought)  
        0 — represents Ungrounded or hallucinated

        Final output must be a single python dictionary as below:
        {'Score': '<0-1>','Justification': '<1-2 sentence explanation for the score>'}"""

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": "GT: " + target + "\n" + "agent's reasoning steps: " + pred}
    ]
)
    return response['choices'][0]['message']['content']



###################### Tool Precision #####################################

def get_precision_score(GT, pred):

    evaluation_prompt = """
        You are an evaluation assistant measuring tool precision in an agent's reasoning step.

        You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought.

        Your task is to assess whether the tool selected by the agent in each reasoning step is the most appropriate tool for the task, by comparing it with the GT.

        Evaluation Guidelines:\\
        If the number of steps in the GT and the model differ:\\
        - Any extra model step beyond the GT steps should receive a score of 0 (hallucinated or unjustified).\\
        - Any GT step that the model omits should also receive a score of 0 (missing reasoning).\\
        The Tool Precision score for the full query is computed as the average of the per-step scores.

        Assess the following per step:

        - Is the tool selected by the agent the same as the one used in the GT?\\
        - Is the tool the most appropriate for the task being attempted?\\

        Scoring Criteria:\\
        1 — represents the selected tool matches the GT tool and is appropriate for the task.\\
        0 — represents the tool does not match the GT or is an inappropriate choice.\\

        Final output must be a single python dictionary as below:
        {'Score': '<0 or 1>','Justification': '<explanation for the score>'}"""


    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": "GT: " + GT + "\n" + "agent's reasoning trace: " + pred}
        # {"role": "user", "content": "task description: " + task_desc + "\n" + "tool options: " + tools + "\n" + "tool chosen: " + tool_selected}
    ]
)
    return response['choices'][0]['message']['content']



###################### Tool Accuracy #####################################

def get_tool_accuray(tool_metadata, reasoning_steps):

    tool_accuracy_prompt = """
        You are an evaluation assistant measuring tool accuracy in an agent's reasoning step.

        You are given the task the agent is trying to accomplish, the tool it uses along with its input and output, and the tool metadata containing a description of the tool's purpose and its expected input/output formats.

        Your task is to assess whether the tool used by the agent was applied correctly in each step by checking:
        - Whether the output format is valid and consistent with the tool's specification (based on tool metadata).
        - Whether the output is relevant and meaningful for completing the step's task.

        Evaluation Guidelines:
        - The Tool Accuracy score for the full query is computed as the average of the per-step scores.

        Assess the following per step:
        - Is the tool's output correctly formatted according to the tool metadata?
        - Is the output meaningful and appropriate for the stated task?

        Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents the output is valid, properly formatted, and clearly relevant to the task.
        0 — represents the output is incorrectly formatted, irrelevant, or unhelpful.

       Final output must be a single python dictionary as below:
        {'Score': '<0-1>','Justification': '<explanation for the score>'}"""


    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": tool_accuracy_prompt},
        # {"role": "user", "content": "tool metadata: " + tool_metadata + "\n" + "tool output: " + tool_output + "\n" + "task goal: " + task_goal}
        {"role": "user", "content": "tool metadata: " + tool_metadata + "\n" + "agent's reasoning steps: " + reasoning_steps }
    ]
)
    return response['choices'][0]['message']['content']


###################### Faithfulness Accuracy #####################################

def get_faithfulness_accuray(GT, reasoning_steps):

    faithfulness_accuracy_prompt = """
        You are an evaluation assistant measuring the Faithfulness Accuracy of an agent's reasoning process.

        You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.
        You are also given the full reasoning trace produced by the agent, including each step's task, selected tool, output, and thought.

        Your task is to assess how faithful the agent's reasoning trace is to the GT — that is, whether the steps follow a logically sound plan that aligns with the structure, intent, and direction of the GT.

        Evaluation Guidelines:
        - Focus on the structure and logical flow of the agent's reasoning steps.
        - Determine whether the steps collectively form a coherent strategy to answer the query.
        - Faithfulness is about consistency with the GT's method, not necessarily correctness of individual steps.

        Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents the reasoning is faithful to the GT: it follows a logically sound plan that mirrors the GT in structure and direction.
        0 — represents the reasoning is not faithful to the GT and lacks logical progression or alignment with the intended plan.

        Final output must be a single python dictionary as below:
        {'Score': '<0-1>','Justification': '< explanation for the score>'}"""




    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": faithfulness_accuracy_prompt},
        {"role": "user", "content": "GT: " + GT + "\n" + "agent's reasoning trace: " + reasoning_steps}
    ]
)
    return response['choices'][0]['message']['content']




###################### Goal Accuracy #####################################

def get_goal_accuray(GT, query_type, final_answer):

    goal_accuracy_prompt = """
        You are an evaluation assistant measuring the Goal Accuracy of a vision-language agent's final answer.

        You are given the agent's final output, the ground truth (GT) final answer, and the type of query — either “objective” or “subjective”.

        Your task is to assess how well the agent's final output matches the ground truth answer, based on the nature of the query.

        Evaluation Guidelines:\\
        - The Goal Accuracy score is computed once per query (not per step).\\
        - Use exact match evaluation for objective queries.\\
        - Use semantic similarity evaluation for subjective queries.\\

        Scoring Criteria:\\
        For objective queries:\\
        1 — The final output matches the GT exactly or is clearly equivalent.\\
        0 — The output is incorrect, incomplete, or unrelated.\\

        For subjective queries:\\
        Score = Cosine similarity between the agent's answer and the GT answer (range: 0 to 1)\\

        if query type is obejctive, the final output must be a single python dictionary as below:: \\
        {'Score': '<0 or 1>','Justification': '<Optional  explanation>'} \\

        if query type is subjective, the final output must be a single python dictionary as below:: \\
        {'Score': <cosine similarity>, 'Justification': '<Optional  explanation>'}
        """



    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": goal_accuracy_prompt},
        {"role": "user", "content": "GT answer: " + GT + "\n" + "query_type: " + query_type + "\n" + "agent's final answer: " + final_answer}
    ]
)
    return response['choices'][0]['message']['content']



###################### ToolUse Accuracy (F1 Score Approximation) #####################################

def get_toolset_accuray(GT, reasoning_steps):

    toolset_accuracy_prompt = """
        You are an evaluation assistant measuring the toolset accuracy of an agent's reasoning process.

        You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

        You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought.

        Your task is to evaluate whether the agent used the correct tools overall by comparing the set of tools it used to the set used in the GT.

        Evaluation Guidelines:

        - If the agent uses tools that are not present in the GT or misses tools that are, it should be penalized.

        - The score reflects how well the agent's toolset aligns with the GT toolset across the full reasoning trace.

        Final output must be a single python dictionary as below:
        {'Score':  <F1 score rounded to 2 decimal places>, 'Justification': '< explanation for the score>'}"""


    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": toolset_accuracy_prompt},
        {"role": "user", "content": "GT: " + GT + "\n" + "agent's reasoning steps: " + reasoning_steps}
    ]
)
    return response['choices'][0]['message']['content']



###################### Step Score #####################################

def get_step_score(reasoning_step):

    step_score_prompt = """
    You are an evaluation assistant measuring the Step Score for each reasoning step in a vision-language agent's process.

    You are given what the agent is trying to accomplish in this step (the task), the agent's rationale explaining why this step is being taken (the thought).

    Your task is to assess whether the thought is logically and clearly connected to the task.

    Evaluation Guidelines:
    - The Step Score for the full query is computed as the average of the per-step scores.

    Assess the following per step:
    - Does the thought demonstrate an understanding of the task's purpose?
    - Is the thought a valid justification for using the tool or progressing to the next step?
    - Is the reasoning coherent, relevant, and logically sound?

    Scoring Criteria: Output the scores between the range 0 and 1 such that:
    1 — represents the thought is clearly connected to the task and provides a well-formed justification.
    0 — represents the thought is disconnected, irrelevant, or nonsensical relative to the task.

    Final output must be a single python dictionary as below:
    {'Score': '<0-1>','Justification': '< explanation for the score>'}"""

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": step_score_prompt},
        {"role": "user", "content": "agent's reasoning step: " + reasoning_step}
    ]
)
    return response['choices'][0]['message']['content']



###################### Context Score #####################################

def get_context_score(GT, reasoning_steps):

    context_score_prompt = """
        You are an evaluation assistant measuring the Context Score of an agent's reasoning step.

        You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

        You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

        Your task is to assess whether the agent's reasoning step is grounded in the input context — and whether that context was effectively used in the reasoning process. Inputs may include image, video, text, audio, or a combination.

        Evaluation Guidelines:

        If the number of steps in the GT and the model differ:

        - Any extra model step beyond the GT steps should receive a score of 0 (hallucinated or unjustified).

        - Any GT step that the model omits should also receive a score of 0 (missing reasoning).

        The Context Score for the full query is computed as the average of the per-step scores.

        Assess the following per step:

        - Does the agent correctly use relevant parts of the input?

        - Is the input used appropriate for the tool selected and the task being attempted?

        - Does the reasoning show effective and meaningful use of the input?

        Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents the agent uses the input fully and appropriately.
        0 — represents the agent ignores or misinterprets the input entirely.

        Final output must be a single python dictionary as below:
        {'Score': '<0 - 1>','Justification': '< explanation for the score>'}"""


    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": context_score_prompt},
        {"role": "user", "content": "GT: " + GT + "\n" + "agent's reasoning_steps: " + reasoning_steps}
    ]
)
    return response['choices'][0]['message']['content']


###################### Clarity Penalty #####################################

def get_clarity_penalty(reasoning_trace):

    clarity_penalty_prompt = """
        You are an evaluation assistant measuring the clarity of an agent's reasoning trace.

        You are given the full reasoning trace produced by the agent, consisting of multiple steps (each with a task, selected tool, input/output, and thought).

        Your task is to identify whether the reasoning is unnecessarily verbose, repetitive, or uninformative — and assign a penalty if applicable.

        Evaluation Guidelines:

        - Penalize the agent if it repeats the same reasoning across multiple steps without adding new information.

        - Penalize if the thoughts are overly wordy, vague, or lacking meaningful content.

        - No penalty if the reasoning is clear, concise, and informative.

        Scoring Criteria (Penalty): Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents the Reasoning is clear and concise.
        0 — represents Highly verbose, redundant, or uninformative reasoning.

        Final output must be a single python dictionary as below:
        {'Score': '<0 - 1>','Justification': '<explanation for the penalty>'}"""

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": clarity_penalty_prompt},
        {"role": "user", "content": "agent's reasoning trace: " + reasoning_trace}
    ]
)
    return response['choices'][0]['message']['content']



###################### Factual Precision #####################################

def get_factual_precision(GT, reasoning_steps):

    factual_accuracy_prompt = """
        You are an evaluation assistant measuring the factual accuracy of the agent's reasoning steps.

        You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

        You are also given the full reasoning trace produced by the agent, including each ste's task, selected tool, tool input/output, and thought.

        Your task is to assess whether the agent introduces any hallucinated, fabricated, or factually incorrect information in its reasoning when compared to the GT.

        Evaluation Guidelines:

        - Compare the model's reasoning steps to the GT to identify factual hallucinations or incorrect claims.

        - Focus on whether the output or thought includes details not supported by the GT.

        - Do not penalize for minor omissions unless they lead to a factual error.

        Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents no factual errors or hallucinations compared to the GT.
        0 — represents major factual errors or hallucinated content.

        Final output must be a single python dictionary as below:
        {'Score': '<0 - 1>','Justification': '< explanation for the score>'}"""

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": factual_accuracy_prompt},
        {"role": "user", "content": "GT: " + GT + "\n" + "agent's reasoning steps: " + reasoning_steps}
    ]
)
    return response['choices'][0]['message']['content']



###################### Semantic Accuracy #####################################

def get_semantic_accuracy(GT, reasoning_steps, final_answer):

    semantic_accuracy_prompt = """You are an evaluation assistant measuring the Semantic Accuracy of an agent's reasoning process.

            You are given a Ground Truth (GT) block containing the original query, a sequence of reasoning steps, and the final answer along with its justification. Each reasoning step includes the task being attempted, the tool used (with its input and output), and the agent's thought process.

            You are also given the full reasoning trace produced by the agent, including each step's task, selected tool and its output, and thought, as well as the agent's final answer and justification.

            Your task is to assess whether the agent's reasoning and final output semantically align with the Ground Truth — that is, whether the agent has covered all the essential parts of the query as demonstrated in the GT.

            Evaluation Guidelines:

            - Compare the agent's reasoning trace and final answer with the GT to check whether all key components of the query are addressed.

            - Credit should be given for meaningful semantic coverage, not superficial similarity.

            - If the model ignores or misunderstands core parts of the GT reasoning or final answer, penalize accordingly.

            Scoring Criteria: Output the scores between the range 0 and 1 such that:
            1 — represents the agent addresses all key components of the query, matching the GT's semantic scope.
            0 — represents the agent's reasoning or answer omits or misrepresents major parts of the query.

        Final output must be a single python dictionary as below:
        {'Score': '<0 - 1>','Justification': '< explanation for the score>'}"""

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": semantic_accuracy_prompt},
        {"role": "user", "content": "GT: " + GT + "\n" + "agent's reasoning steps: " + reasoning_steps + "\n" + "agent's final answer: " + final_answer}
    ]
)
    return response['choices'][0]['message']['content']



###################### Coherence Accuracy #####################################

def get_coherence_accuracy(reasoning_steps):

    instruct_prompt = (
    "You are given:\n"
    "- Full reasoning trace.\n"
    "Task:\n"
    "Check whether the reasoning steps are logically connected without contradictions or missing transitions.\n"
    "Score:\n"
    "- 1: Fully coherent.\n"
    "- 0.5: Minor coherence gaps.\n"
    "- 0: Major logical gaps.\n"
    "Output score (0, 0.5, or 1) and a one-line explanation."
)



    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": instruct_prompt},
        {"role": "user", "content": "agent's reasoning trace: " + reasoning_steps}
    ]
)
    return response['choices'][0]['message']['content']



###################### Reward Score (Self-correction Ability) #####################################

def get_reward_score(query, reasoning_steps):

    reward_score_prompt = """
        You are an evaluation assistant measuring the Reward Score of a multi-modal agent — that is, its ability to recognize and correct its own reasoning mistakes.

        You are given the full reasoning trace produced by the agent, including each step's task, selected tool, input/output, and thought.

        You are also given the query as a reference for which the agent has to produce the final answer using the reasoning steps.

        Your task is to assess whether the agent demonstrates self-correction — meaning that it identifies a mistake in its reasoning and actively corrects it to move toward a more accurate solution.

        Evaluation Guidelines:
        - Look for signs that the agent realized it made a wrong assumption, took an incorrect action, or misinterpreted something.
        - Then check whether it explicitly adjusted or revised its reasoning to fix the issue.
        - Self-correction should happen within the reasoning trace itself (not just the final answer).

        Scoring Criteria: Output the scores between the range 0 and 1 such that:
        1 — represents the agent clearly identifies a mistake and corrects it through subsequent steps.
        0 — represents the agent does not acknowledge the mistake or continues the incorrect reasoning path.

        Final output must be a single python dictionary as below:
        {'Score': '<0 - 1>' ,'Justification': '< explanation for the score>}
        """


    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": reward_score_prompt},
        {"role": "user", "content": "query: " + query + "\n" + "agent's reasoning trace: " + reasoning_steps}
    ]
)
    return response['choices'][0]['message']['content']