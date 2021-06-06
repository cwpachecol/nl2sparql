import json
import os

import requests, json, re, operator
import sys
from parsers.lc_quad20 import LC_Qaud20

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

    start_record = 0
    end_record = 0
    # show_json(filename='data/lcquad10/data.json', start_rec=start_record, end_rec=end_record)

    with open('data/lcquad20/train.json', 'r', encoding='utf-8') as f:
        train = json.load(f)

    with open('data/lcquad20/test.json', 'r', encoding='utf-8') as f:
        test = json.load(f)

    data = train + test

    if end_record == 0:
        end_record = len(data)

    print('data len: ', len(data))

    with open("data/lcquad20/data.json", "w") as write_file:
        json.dump(data, write_file)

    # ds = LC_Qaud20(path="./data/lcquad20/data.json", sparql_field="sparql_wikidata")
    ds = LC_Qaud20(path="./data/lcquad20/data.json", sparql_field="sparql_dbpedia18")
    tmp = []
    for idx, qapair in enumerate(prepare_dataset(ds).qapairs[start_record: end_record]):
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
        if idx % 100 == 0:
            print(f"No: {idx} \n row: {raw_row}")
        # write_json(raw_row, filename='data/lcquad20/linked_answer.json')

            with open('data/lcquad20/linked_answer_' + str(start_record) + '_to_' + str(idx) + '.json', 'w') as jsonFile:
                json.dump(tmp, jsonFile)

    print('data len: ', len(tmp))