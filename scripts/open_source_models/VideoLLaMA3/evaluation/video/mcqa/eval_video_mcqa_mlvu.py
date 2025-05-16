import json
import argparse
from tabulate import tabulate


def main():
    args = parse_args()
    res = [eval(x.strip()) for x in open(args.pred_path, 'r').readlines()]

    task_acc = {}
    # task_acc = {x: [] for x in task_types}
    acc = []
    task_typs = []
    for i, x in enumerate(res):
        value = 1
        if x['pred'] != x['gt']:
            value = 0
        acc.append(value)
        # print(x['task_type'])
        task_typs.append(x['task_type'])
        if x['task_type'] not in task_acc:
            task_acc[x['task_type']] = [value]
        else:
            task_acc[x['task_type']].append(value)
    acc = sum(acc) * 100 / len(acc)
    task_acc = {x: sum(task_acc[x]) * 100 / len(task_acc[x]) for x in task_acc}
    print(f"{args.pred_path}: {acc:.2f}")
    task_names = list(task_acc.keys())
    
    table_data = []
    for i in range(len(task_names) // 4):
        row_task_names = task_names[i * 4: (i + 1) * 4]
        row_task_acc = [task_acc[x] for x in row_task_names]
        table_data.append(row_task_names)
        table_data.append(row_task_acc)
    table_data.append(task_names[(i + 1) * 4:])
    table_data.append([task_acc[x] for x in task_names[(i + 1) * 4:]])
    print(tabulate(table_data, floatfmt=".2f"), '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video captioning.")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
