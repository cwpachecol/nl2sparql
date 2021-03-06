import json
import os
import argparse
import requests, json, re, operator
import sys
from parsers.lc_quad20 import LC_Quad20
import shutil

def show_json(filename='data.json', start_rec=0, end_rec=0):
    with open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        print(f"len of {filename}: {len(data)}")
        json_file.close()
    nr = start_rec
    if end_rec == 0:
        end_rec = len(data)
    else:
        end_rec += 1
    for rec in data[start_rec:end_rec]:
        print(f"record no: {nr}")
        print(rec)
        print(rec['_id'])
        print(rec['corrected_question'])
        nr += 1
def append_json(new_data, filename='data.json'):
    if not os.path.isfile(filename):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(new_data, file, indent=4)
            file.close()
    else:
        with open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            json_file.close()
            temp = data[0]
            temp.append(new_data)
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(temp, file, indent=4)
                file.close()




def write_json(new_data, filename='data.json'):

    if not os.path.isfile(filename):
        with open(filename, 'w', encoding='utf-8') as file:
            # json.dump(new_data, file, indent=4)
            # json.dump(new_data, file, indent=4)
            file.write(json.dumps(new_data))
            file.write(",")
            file.close()
    else:
        with open(filename, 'a', encoding='utf-8') as file:
            # First we load existing data into a dict.
            # file_data = json.load(file)
            # Join new_dat3a with file_data
            file.write(json.dumps(new_data))
            file.write(",")
            # print(file_data)
            # file_data.update(new_data)

            # Sets file's current position at offset.
            # file.seek(0)
            # # convert back to json.
            # json.dump(file_data, file, indent=4)
            file.close()

def prepare_dataset(ds):
    ds.load()
    ds.parse()
    return ds

def ask_query(uri):
    if uri == "<https://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
        return 200, json.loads("{\"boolean\": \"True\"}")
    uri = uri.replace("https://", "http://")
    return query(u'ASK WHERE {{ {} ?u ?x }}'.format(uri))

def query(q):
    q = q.replace("https://", "http://")
    payload = (
        ('query', q),
        ('format', 'application/json'))

    r = requests.get('http://dbpedia.org/sparql', params=payload)
    return r.status_code, r.json()

def has_answer(t):
    if "results" in t and len(t["results"]["bindings"]) > 0:
        return True
    if "boolean" in t:
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drive directory from colab')
    parser.add_argument('--copy_to_drive', default=True, action='store_true',
                        help='Enable capacity to save in drive')
    parser.add_argument('--drive_directory', default='/content/gdrive/MyDrive/Colab Notebooks/nl2sparql/data/lcquad20/',
                        help='directory in drive to save json files')
    parser.add_argument('--start_index', default=0, type=int,
                        help='start index of dataset')
    parser.add_argument('--end_index', default=0, type=int,
                        help='end index of dataset')

    args = parser.parse_args()
    start_index = args.start_index
    end_index = args.end_index
    # copy_to_drive = args.copy_to_drive
    copy_to_drive = False
    drive_directory = args.drive_directory
    # show_json(filename='data/lcquad10/data.json', start_rec=start_record, end_rec=end_record)

    with open('data/lcquad20/train.json', 'r', encoding='utf-8') as f:
        train = json.load(f)

    with open('data/lcquad20/test.json', 'r', encoding='utf-8') as f:
        test = json.load(f)

    data = train + test

    if end_index == 0:
        end_index = len(data)

    print('data len: ', len(data))

    with open("data/lcquad20/data.json", "w") as write_file:
        json.dump(data, write_file)

    # ds = LC_Quad20(path="./data/lcquad20/data.json", sparql_field="sparql_wikidata")
    ds = LC_Quad20(path="./data/lcquad20/data.json", sparql_field="sparql_dbpedia18")
    tmp = []
    actual_index = start_index
    for idx, qapair in enumerate(prepare_dataset(ds).qapairs[start_index: end_index]):
        raw_row = dict()
        raw_row["id"] = qapair.id.__str__()
        raw_row["question"] = qapair.question.text
        raw_row["sparql_query"] = qapair.sparql.query
        try:
            r = query(qapair.sparql.query)
            raw_row["answers"] = r[1]
        except Exception as e:
            raw_row["answers"] = []

        tmp.append(raw_row)
        actual_index += 1
        if ((actual_index > 0) and (actual_index % 100 == 0)) or actual_index >= end_index:
            print(f"No: {idx} \n row: {raw_row}")
        # write_json(raw_row, filename='data/lcquad20/linked_answer.json')

            with open('data/lcquad20/linked_answer_' + str(start_index) + '_to_' + str(actual_index) + '.json', 'w') as jsonFile:
                json.dump(tmp, jsonFile)
                jsonFile.close()
            if copy_to_drive:
                shutil.copy('data/lcquad20/linked_answer_' + str(start_index) + '_to_' + str(actual_index) + '.json', drive_directory)

    print('data len: ', len(tmp))