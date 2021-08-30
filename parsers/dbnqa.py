import json, re
from common.container.qapair import QApair
from common.container.uri import Uri
from kb.dbpedia import DBpedia
from parsers.answerparser import AnswerParser
# ./data/LC-QUAD/data_v8.json
# {"verbalized_question": "Who are the <comics characters> whose <painter> is <Bill Finger>?",
#  "_id": "f0a9f1ca14764095ae089b152e0e7f12",
#  "sparql_template_id": 301,
#  "sparql_query": "SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/ontology/creator> <http://dbpedia.org/resource/Bill_Finger>  . ?uri <https://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/ComicsCharacter>}",
#  "corrected_question": "Which comic characters are painted by Bill Finger?"}
class DBNQA:
    # def __init__(self, path="./data/LC-QUAD/data_v8.json"):
    def __init__(self, path="./data/dbnqa/data.json"):
        self.raw_data = []
        self.qapairs = []
        self.path = path
        self.parser = DBNQAParser()

    def load(self):
        with open(self.path) as data_file:
            self.raw_data = json.load(data_file)

    def parse(self):
        parser = DBNQAParser()
        for raw_row in self.raw_data:
            sparql_query = raw_row["sparql"].replace("DISTINCT COUNT(", "COUNT(DISTINCT ")
            self.qapairs.append(
                QApair(raw_row["question"], [], sparql_query, raw_row, raw_row["id"], self.parser))

    def print_pairs(self, n=-1):
        for item in self.qapairs[0:n]:
            print(item)
            print("--")


class DBNQAParser(AnswerParser):
    def __init__(self):
        super(DBNQAParser, self).__init__(DBpedia(one_hop_bloom_file="./data/blooms/spo1.bloom"))

    def parse_question(self, raw_question):
        return raw_question

    def parse_sparql(self, raw_query):
        uris = [Uri(raw_uri, DBpedia.parse_uri) for raw_uri in re.findall('<[^>]*>', raw_query)]

        return raw_query, True, uris

    def parse_answerset(self, raw_answerset):
        return []

    def parse_answerrow(self, raw_answerrow):
        return []

    def parse_answer(self, answer_type, raw_answer):
        return "", None
