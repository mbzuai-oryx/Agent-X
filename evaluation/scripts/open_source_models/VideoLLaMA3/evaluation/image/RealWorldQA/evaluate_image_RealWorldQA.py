"""Parse and Evalate"""
import json
from argparse import ArgumentParser


def evaluate_exact_match_accuracy(results):
    scores = []
    for question_id in results.keys():
        elem = results[question_id]

        ann = elem['annotation'].strip().lower()

        pred = elem['prediction'].strip().lower()

        if pred[-1] == '.':
            pred = pred[:-1]

        score = 1.0 if pred == ann else 0.0

        # print(ann, pred, score)

        scores.append(score)
    return sum(scores) / len(scores)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results-file', type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)

    args = parser.parse_args()

    # read results
    if args.num_chunks > 1:
        results = {}
        for _idx in range(args.num_chunks):
            file = args.results_file.replace('.json', f'_{args.num_chunks}_{_idx}.json')
            results.update(read_json(file))
    else:
        results = read_json(args.results_file)

    acc = evaluate_exact_match_accuracy(results)

    print(f'overall accuracy: {acc}')

