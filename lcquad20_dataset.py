import json
import requests, json, re, operator
import sys
from parsers.lc_quad20 import LC_Qaud20
import pandas as pd


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

    with open('data/LC-QUAD20/train.json', 'r', encoding='utf-8') as f:
        train = json.load(f)

    with open('data/LC-QUAD20/test.json', 'r', encoding='utf-8') as f:
        test = json.load(f)

    data = train + test
    print('data len: ', len(data))

    with open("data/LC-QUAD20/data.json", "w") as write_file:
        json.dump(data, write_file)

    # ds = LC_Qaud20(path="./data/LC-QUAD20/data.json", sparql_field="sparql_wikidata")
    ds = LC_Qaud20( path="./data/LC-QUAD20/data.json", sparql_field="sparql_dbpedia18")
    tmp = []
    for idx, qapair in enumerate(prepare_dataset(ds).qapairs):
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
    with open('data/LC-QUAD20/linked_answer.json', 'w') as jsonFile:
        json.dump(tmp, jsonFile)

    print('data len: ', len(tmp))
