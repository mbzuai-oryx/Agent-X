import json
import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        file = f.readlines()
    json_file = [json.loads(i.strip()) for i in file]
    with open(sys.argv[1], 'w') as f:
        json.dump(json_file, f)