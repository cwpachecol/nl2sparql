import json
import os


def merge_json_files():
    with open('data/lcquad20/linked_answer_0_to_1400.json', 'r', encoding='utf-8') as f:
        first_file = json.load(f)

    with open('data/lcquad20/linked_answer_1401_to_30226.json', 'r', encoding='utf-8') as f:
        second_file = json.load(f)

    merge_file = first_file + second_file

    with open("data/lcquad20/linked_answer.json", "w") as write_file:
        json.dump(merge_file, write_file)


if __name__ == "__main__":
    merge_json_files()
    with open('data/lcquad20/data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    with open('data/lcquad20/linked_answer.json', 'r', encoding='utf-8') as f:
        linked_answer = json.load(f)
    print(len(linked_answer))