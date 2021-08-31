import json
import re
from common.container.qapair import QApair
from common.container.uri import Uri
from common.container.uris import URIs
from kb.dbpedia import DBpedia
from parsers.answerparser import AnswerParser


class DBNQA_Linked:
    def __init__(self, path="./data/dbnqa/linked.json"):
        self.raw_data = []
        self.qapairs = []
        self.path = path
        self.parser = DBNQA_LinkedParser()

    def load(self):
        with open(self.path) as data_file:
            self.raw_data = json.load(data_file)

    def parse(self):
        for raw_row in self.raw_data:
            # print(raw_row)
            self.qapairs.append(
                QApair(raw_row["question"], raw_row.get("answers"), raw_row["sparql"], raw_row, raw_row["id"],
                       self.parser))

    def print_pairs(self, n=-1):
        for item in self.qapairs[0:n]:
            # print(item.raw_row)
            # print(item.question.raw_question)
            # print(item.question.text)
            print("-" * 30)
            print(f"sparql.raw_query: {item.sparql.raw_query}")
            print(f"sparql.query: {item.sparql.query}")
            print(f"sparql.supported: {item.sparql.supported}")
            print(f"sparql.uris.raw_uri: {[uri.raw_uri for uri in item.sparql.uris]}")
            print(f"sparql.uris.type: {[uri.uri_type for uri in item.sparql.uris]}")

            print(f"sparql.where_clause: {item.sparql.where_clause}")
            print(f"sparql.where_clause_template: {item.sparql.where_clause_template}")
            print("="*50)

class DBNQA_LinkedParser(AnswerParser):
    def __init__(self):
        super(DBNQA_LinkedParser, self).__init__(DBpedia(one_hop_bloom_file="./data/blooms/spo1.bloom"))

    def parse_question(self, raw_question):
        return raw_question

    def parse_answerset(self, raw_answers):
        return self.parse_queryresult(raw_answers)

    def parse_sparql(self, raw_query):
        raw_query = raw_query.replace("https://", "http://")
        uris = URIs([Uri(raw_uri, self.kb.parse_uri) for raw_uri in re.findall('(<[^>]*>|\?[^ ]*)', raw_query)])
        return raw_query, True, uris
