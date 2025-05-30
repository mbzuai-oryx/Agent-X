import argparse
import json
import math
import sys

sys.path.append('./')
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from videollama2 import mm_infer, model_init
from videollama3 import disable_torch_init


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default="./example_outputs/qwen_vl/total_val_output.json", help="The path to model output file.")
    parser.add_argument("--num-chunks", type=int, default=1)

    args = parser.parse_args()

    OCRBench_score = {"Regular Text Recognition":0, "Irregular Text Recognition":0, "Artistic Text Recognition":0,
                      "Handwriting Recognition":0, "Digit String Recognition":0, "Non-Semantic Text Recognition":0,
                      "Scene Text-centric VQA":0, "Doc-oriented VQA":0, "Key Information Extraction":0,
                      "Handwritten Mathematical Expression Recognition":0}

    AllDataset_score = {"IIIT5K":0, "svt":0, "IC13_857":0, "IC15_1811":0, "svtp":0,
                        "ct80":0, "cocotext":0, "ctw":0, "totaltext":0, "HOST":0,
                        "WOST":0, "WordArt":0, "IAM":0, "ReCTS":0, "ORAND":0, "NonSemanticText":0,
                        "SemanticText":0, "STVQA":0, "textVQA":0, "ocrVQA":0, "ESTVQA":0, "ESTVQA_cn":0,
                        "docVQA":0, "infographicVQA":0, "ChartQA":0, "ChartQA_Human":0, "FUNSD":0,
                        "SROIE":0,"POIE":0,"HME100k":0}

    num_all = {"IIIT5K":0, "svt":0, "IC13_857":0, "IC15_1811":0, "svtp":0, "ct80":0,
               "cocotext":0, "ctw":0, "totaltext":0, "HOST":0, "WOST":0, "WordArt":0,
               "IAM":0, "ReCTS":0, "ORAND":0, "NonSemanticText":0, "SemanticText":0,
               "STVQA":0, "textVQA":0, "ocrVQA":0, "ESTVQA":0, "ESTVQA_cn":0, "docVQA":0,
               "infographicVQA":0, "ChartQA":0, "ChartQA_Human":0, "FUNSD":0, "SROIE":0,
               "POIE":0, "HME100k":0}

    if args.num_chunks > 1:
        results = []
        for _idx in range(args.num_chunks):
            file = args.output_path.replace('.json', f'_{args.num_chunks}_{_idx}.json')
            results += json.load(open(file))
    else:
        results = json.load(open(args.output_path))

    for i in range(len(results)):
        data_type = results[i]["type"]
        dataset_name = results[i]["dataset_name"]
        answers = results[i]["answers"]
        if results[i].get('predict',0)==0:
            continue
        predict = results[i]['predict']
        results[i]['result'] = 0
        if dataset_name == "HME100k":
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n"," ").replace(" ","")
                    predict = predict.strip().replace("\n"," ").replace(" ","")
                    if answer in predict:
                        results[i]['result'] = 1
            else:
                answers = answers.strip().replace("\n"," ").replace(" ","")
                predict = predict.strip().replace("\n"," ").replace(" ","")
                if answers in predict:
                    results[i]['result'] = 1
        else:
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n"," ")
                    predict = predict.lower().strip().replace("\n"," ")
                    if answer in predict:
                        results[i]['result'] = 1
            else:
                answers = answers.lower().strip().replace("\n"," ")
                predict = predict.lower().strip().replace("\n"," ")
                if answers in predict:
                    results[i]['result'] = 1

    # save_json(results, args.save_file)

    if len(results)==1000:
        for i in range(len(results)):
            if results[i].get("result",100)==100:
                continue
            OCRBench_score[results[i]['type']] += results[i]['result']
        recognition_score = OCRBench_score['Regular Text Recognition']+OCRBench_score['Irregular Text Recognition']+OCRBench_score['Artistic Text Recognition']+OCRBench_score['Handwriting Recognition']+OCRBench_score['Digit String Recognition']+OCRBench_score['Non-Semantic Text Recognition']
        Final_score = recognition_score+OCRBench_score['Scene Text-centric VQA']+OCRBench_score['Doc-oriented VQA']+OCRBench_score['Key Information Extraction']+OCRBench_score['Handwritten Mathematical Expression Recognition']
        print("###########################OCRBench##############################")
        print(f"Text Recognition(Total 300):{recognition_score}")
        print("------------------Details of Recognition Score-------------------")
        print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
        print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
        print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
        print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
        print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
        print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
        print("----------------------------------------------------------------")
        print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
        print("----------------------------------------------------------------")
        print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
        print("----------------------------------------------------------------")
        print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
        print("----------------------------------------------------------------")
        print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
        print("----------------------Final Score-------------------------------")
        print(f"Final Score(Total 1000): {Final_score}")
    else:
        for i in range(len(results)):
            num_all[results[i]['dataset_name']] += 1
            if results[i].get("result",100)==100:
                continue
            AllDataset_score[results[i]['dataset_name']] += results[i]['result']
        for key in AllDataset_score.keys():
            print(f"{key}: {AllDataset_score[key]/float(num_all[key])}")
