import argparse
import json
import os
import re

import sys
sys.path.append('./')

import pandas as pd
# !pip install python-Levenshtein
from Levenshtein import distance
from tqdm import tqdm

# OpenAI
import openai
from openai import AzureOpenAI
# openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key)
# test

from evaluation.image.MathVista.ext_ans import demo_prompt

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def get_chat_response(promot, n=1, patience=10000000,
 sleep_time=0):
    messages = [
        {"role": "user", "content": promot},
    ]
    # print("I am here")
    while patience > 0:
        patience -= 1
        try:
            response = interaction(client, messages)
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce promot size")
                # reduce input prompt and keep the tail
                new_size = int(len(promot) * 0.9)
                new_start = len(promot) - new_size
                promot = promot[new_start:]
                messages = [
                    {"role": "user", "content": promot},
                ]
                
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""


def init():
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

    return client


def interaction(client, message_text):
    completion = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYNAME"),
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion


def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass
    
    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass
    
    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass
        
    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = get_chat_response(full_prompt, patience=10)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""


def extract_all_answers(args):

    # read results
    if args.num_chunks > 1:
        results = {}
        for _idx in range(args.num_chunks):
            file = args.results_file.replace('.json', f'_{args.num_chunks}_{_idx}.json')
            results.update(read_json(file))
    else:
        results = read_json(args.results_file)

    test_pids = list(results.keys())
    print("Number of testing problems:", len(test_pids))

    label = args.response_label

    test_num = len(test_pids)
    print("Number of problems to run:", test_num)

    # tqdm, enumerate results
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]

        extraction  = extract_answer(response, problem, args.quick_extract)
        results[pid]['extraction'] = extraction

        if i % args.save_every == 0 or i == test_num - 1:
            print(f"Saving results to {args.output_file}...")
            save_json(results, args.output_file)
            print(f"Results saved.")

    return results


def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        options = [chr(ord('A') + i) for i in range(len(choices))]

        if extraction in options:
            # convert option letter to text, e.g. "A" -> "text"
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            # select the most similar option
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == 'integer':
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None

    elif answer_type == 'float':
        try:
            extraction = str(round(float(extraction), int(precision)))
        except:
            extraction = None

    elif answer_type == 'list':
        try:
            extraction = str(extraction)
        except:
            extraction = None

    return extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False



def get_acc_with_contion(res_pd, key, value):
    if key == 'skills':
        # if value in res_pd[key]:
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return len(correct_pd), len(total_pd), acc


def calculate_scores(args, results):
    full_pids = list(results.keys())
    print("Number of testing problems:", len(full_pids))

    ## [1] Evaluate if the prediction is true or false
    print("\nEvaluating the predictions...")
    update_json_flag = False

    for pid in full_pids:
        problem = results[pid]
        # print(problem)

        choices = problem['choices']
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        precision = problem['precision']
        extraction = problem['extraction']


        answer = problem['answer']

        # normalize the extracted answer to match the answer type
        prediction = normalize_extracted_answer(extraction, choices, question_type, answer_type, precision)

        # verify the prediction is true or false
        true_false = safe_equal(prediction, answer)

        # update the problem
        if "true_false" not in problem:
            update_json_flag = True

        elif true_false != problem['true_false']:
            update_json_flag = True

        if "prediction" not in problem:
            update_json_flag = True

        elif prediction != problem['prediction']:
            update_json_flag = True

        problem['prediction'] = prediction
        problem['true_false'] = true_false

    # save the updated json
    if update_json_flag:
        print("\n!!!Some problems are updated.!!!")
        print(f"\nSaving {args.output_file}...")
        save_json(results, args.output_file)

    ## [2] Calculate the average accuracy
    total = len(full_pids)
    correct = 0
    for pid in full_pids:
        if results[pid]['true_false']:
            correct += 1
    accuracy = str(round(correct / total * 100, 2))
    print(f"\nCorrect: {correct}, Total: {total}, Accuracy: {accuracy}%")

    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    ## [3] Calculate the fine-grained accuracy scores

    # merge the 'metadata' attribute into the data
    for pid in results:
        results[pid].update(results[pid].pop('metadata'))

    # convert the data to a pandas DataFrame
    df = pd.DataFrame(results).T

    print(len(df))
    print("Number of test problems:", len(df))
    # assert len(df) == 1000 # Important!!!

    # asign the target keys for evaluation
    target_keys = ['question_type', 'answer_type', 'language', 'source', 'category', 'task', 'context', 'grade', 'skills']

    for key in target_keys:
        print(f"\nType: [{key}]")
        # get the unique values of the key
        if key == 'skills':
            # the value is a list
            values = []
            for i in range(len(df)):
                values += df[key][i]
            values = list(set(values))
        else:
            values = df[key].unique()
        #print(values)

        # calculate the accuracy for each value
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_contion(df, key, value)
            if total > 0:
                print(f"[{value}]: {acc}% ({correct}/{total})")
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}

        # sort the scores by accuracy
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]['accuracy']), reverse=True))

    # save the scores
    # scores_file = os.path.join(args.output_dir, args.score_file)
    print(f"\nSaving {args.scores_file}...")
    save_json(scores, args.scores_file)
    print("\nDone!")

    # [4] Calculate the score gains over random guess
    # if args.caculate_gain:
    #     random_file = os.path.join(args.output_dir, args.random_file)
    #     random_scores = json.load(open(random_file))

    #     print("\nCalculating the score gains...")
    #     for key in scores:
    #         if key == 'average':
    #             gain = round(float(scores[key]['accuracy']) - float(random_scores[key]['accuracy']), 2)
    #             scores[key]['acc_gain'] = gain
    #         else:
    #             for sub_key in scores[key]:
    #                 gain = round(float(scores[key][sub_key]['accuracy']) - float(random_scores[key][sub_key]['accuracy']), 2)
    #                 scores[key][sub_key]['acc_gain'] = str(gain)

    #     # save the score gains
    #     print(f"\nSaving {scores_file}...")
    #     save_json(scores, scores_file)
    #     print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # # model
    # parser.add_argument('--llm_engine', type=str, default='gpt-4-0613', help='llm engine',
    #                     choices = ['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613'])
    # parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    # parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument('--scores-file', type=str, default='scores.json')
    # parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    parser.add_argument("--api-key", required=True, type=str, help="Azure Openai API key.")
    parser.add_argument("--api-endpoint", required=True, type=str, help="Azure Openai API endpoint.")
    parser.add_argument("--api-deployname", required=True, type=str, help="Azure Openai API deployname.")
    args = parser.parse_args()

    # Set the OpenAI API key.
    os.environ["AZURE_OPENAI_KEY"] = args.api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = args.api_endpoint
    os.environ["AZURE_OPENAI_DEPLOYNAME"] = args.api_deployname
    client = init()

    results = extract_all_answers(args)

    calculate_scores(args, results)
