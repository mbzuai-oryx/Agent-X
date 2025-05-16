"""Parse and Evalate"""
import json
from argparse import ArgumentParser


def evaluate_exact_match_accuracy(results):
    scores = []
    for question_id in results.keys():
        elem = results[question_id]
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
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

