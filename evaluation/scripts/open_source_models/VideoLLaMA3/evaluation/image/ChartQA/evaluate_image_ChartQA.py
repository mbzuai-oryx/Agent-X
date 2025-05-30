"""Parse and Evalate"""
import json
import os
from argparse import ArgumentParser
from typing import Optional


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(results):
    scores = []
    for question_id in results.keys():
        elem = results[question_id]
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(ann, elem['answer'].strip())
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
    parser.add_argument('--test-file', type=str, default='test.jsonl')

    args = parser.parse_args()

    results_file = args.results_file.replace('.json', args.test_file).replace('jsonl', 'json')

    # read results
    if args.num_chunks > 1:
        results = {}
        for _idx in range(args.num_chunks):
            file = results_file.replace('.json', f'_{args.num_chunks}_{_idx}.json')
            results.update(read_json(file))
    else:
        results = read_json(results_file)

    acc = evaluate_relaxed_accuracy(results)

    print(f'overall accuracy: {args.test_file}, {acc}')

