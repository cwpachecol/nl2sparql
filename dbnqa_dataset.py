import json
import os
import requests, json, re, operator
import sys
from parsers.dbnqa import DBNQA
import csv
from config_dbnqa import parse_args

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
    # Argumentos de configuracion general
    global args
    args = parse_args()

    data_dir = args.data
    # global logger

    data_csv_file = os.path.join(data_dir, 'data.csv')
    if not os.path.isfile(data_csv_file):
        with open(os.path.join(data_dir, 'data.csv'), 'w') as file3:
            with open(os.path.join(data_dir, 'data.en'), 'r', encoding='utf-8') as file1:
                with open(os.path.join(data_dir, 'data.sparql'), 'r', encoding='utf-8') as file2:
                    data_writer = csv.writer(file3, delimiter=',', lineterminator='\n')
                    qs_list = []
                    for line1, line2 in zip(file1, file2):
                        question_text = line1.strip()
                        sparql_text = line2.strip().replace('brack_open', '{').replace('brack_close', '}').replace('sep_dot', '.').\
                            replace('attr_open ', '(').replace(' attr_close', ')').replace('_ ', '_').replace('( ', '(').\
                            replace(' )', ')').replace(' var_a ', ' ?var_a ').replace(' var_b ', ' ?var_b ').replace(' var_uri ', ' ?var_uri ').\
                            replace('var_amath_gtvar_b', '?var_a > ?var_b')
                        # sparql_text = sparql_text.
                        sparql_text = sparql_text.replace('dbr_','http://dbpedia.org/resource/').\
                            replace('dbo_', 'http://dbpedia.org/ontology/').replace('dbp_', 'http://dbpedia.org/property/').\
                            replace('rdf_', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#/').replace('rdfs_', 'http://www.w3.org/TR/2014/PER-rdf-schema-20140109/')
                        # file3.write((line1.strip(), line2.strip()))
                        # print(line1.strip(), line2.strip(), file=file3)
                        new_sparql_text = ""
                        for word in sparql_text.split():
                            # print(word)
                            temp_word = ''
                            for item in ['http://dbpedia.org/resource/', 'http://dbpedia.org/ontology/',
                                         'http://dbpedia.org/property/', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#']:
                                if word.rfind(item) != -1:
                                    temp_word = '<' + word + '>'
                                    break

                            if temp_word == '':
                                new_sparql_text = new_sparql_text + word + " "
                            else:
                                new_sparql_text = new_sparql_text + temp_word + " "

                        qs_list.append([question_text.encode('ascii', 'ignore').decode('ascii'), new_sparql_text.encode('ascii', 'ignore').decode('ascii')])

                    data_writer.writerow(["question", "sparql"])
                    for i in range(len(qs_list)):
                        data_writer.writerow(qs_list[i])

    data_json_file = os.path.join(data_dir, 'data.json')

    if not os.path.isfile(data_json_file):
        with open(os.path.join(data_dir, 'data.csv'), 'r', encoding='utf-8') as csv_file:
            csv_data = csv.DictReader(csv_file, delimiter=',')
            data = []

            for idx, row in enumerate(csv_data):
                data_aux = {}
                data_aux['id'] = idx + 1
                data_aux['question'] = row['question']
                data_aux['sparql'] = row['sparql']
                data.append(data_aux)
            # print('data len: ', len(csv_data))
            with open("data/dbnqa/data.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

    ds = DBNQA(os.path.join(data_dir, 'data.json'))
    ds.load()
    ds.parse()

    linked_answer_list = []
    start_element = 1

    linked_answer_json_file = os.path.join(data_dir, 'linked_answer.json')

    if os.path.isfile(linked_answer_json_file):
        with open(os.path.join(data_dir, 'linked_answer.json'), 'r', encoding='utf-8') as la_json_file:
            linked_answer_list = json.load(la_json_file)
        la_json_file.close()

        if len(linked_answer_list) > 0 and linked_answer_list[-1].get('id') is not None:
            start_element = int(linked_answer_list[-1].get('id')) + 1

    # print(f'start_element: {start_element.__str__() }')

    for qapair in ds.qapairs[start_element:]:
        raw_row = dict()
        # print(f'Id: {qapair.id.__str__()}')
        raw_row["id"] = qapair.id.__str__()
        raw_row["question"] = qapair.question.text
        raw_row["sparql"] = qapair.sparql.query
        try:
            r = query(qapair.sparql.query)
            raw_row["answers"] = r[1]
        except Exception as e:
            raw_row["answers"] = []

        linked_answer_list.append(raw_row)

        if qapair.id % 10 == 0:
            print(f'Id: {qapair.id.__str__()}')
            with open(os.path.join(data_dir, 'linked_answer.json'), 'w') as jsonFile:
                json.dump(linked_answer_list, jsonFile)
            jsonFile.close()

    with open(os.path.join(data_dir, 'linked_answer.json'), 'w') as jsonFile:
        json.dump(linked_answer_list, jsonFile)
    jsonFile.close()

    print('data len: ', len(linked_answer_list))
