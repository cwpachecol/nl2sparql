from parsers.lc_quad20_linked import LC_Qaud20_Linked
from parsers.lc_quad20 import LC_Qaud20Parser
from common.container.sparql import SPARQL
from common.container.answerset import AnswerSet
from common.graph.graph import Graph
from common.utility.stats import Stats
from common.query.querybuilder import QueryBuilder
import common.utility.utility as utility
from linkers.goldLinker import GoldLinker
from linkers.earl import Earl
from learning.classifier.svmclassifier import SVMClassifier
import json
import argparse
import logging
import sys
import os
import numpy as np


def safe_div(x, y):
    if y == 0:
        return None
    return x / y


def qg(linker, kb, parser, qapair, force_gold=True):
    logger.info(qapair.sparql)
    logger.info(qapair.question.text)

    h1_threshold = 9999999

    # Get Answer from KB online
    status, raw_answer_true = kb.query(qapair.sparql.query.replace("https", "http"))
    answerset_true = AnswerSet(raw_answer_true, parser.parse_queryresult)
    qapair.answerset = answerset_true

    ask_query = "ASK " in qapair.sparql.query
    count_query = "COUNT(" in qapair.sparql.query
    sort_query = "order by" in qapair.sparql.raw_query.lower()
    entities, ontologies = linker.do(qapair, force_gold=force_gold)

    double_relation = False
    relation_uris = [u for u in qapair.sparql.uris if u.is_ontology() or u.is_type()]
    if len(relation_uris) != len(set(relation_uris)):
        double_relation = True
    else:
        double_relation = False

    print('ask_query: ', ask_query)
    print('count_query: ', count_query)
    print('double_relation: ', double_relation)

    if entities is None or ontologies is None:
        return "-Linker_failed", []

    graph = Graph(kb)
    queryBuilder = QueryBuilder()

    logger.info("start finding the minimal subgraph")

    graph.find_minimal_subgraph(entities, ontologies, double_relation=double_relation, ask_query=ask_query,
                                sort_query=sort_query, h1_threshold=h1_threshold)
    logger.info(graph)
    wheres = queryBuilder.to_where_statement(graph, parser.parse_queryresult, ask_query=ask_query,
                                             count_query=count_query, sort_query=sort_query)

    output_where = [{"query": " .".join(item["where"]), "correct": False, "target_var": "?u_0"} for item in wheres]
    for item in list(output_where):
        logger.info(item["query"])
    if len(wheres) == 0:
        return "-without_path", output_where
    correct = False

    for idx in range(len(wheres)):
        where = wheres[idx]

        if "answer" in where:
            answerset = where["answer"]
            target_var = where["target_var"]
        else:
            target_var = "?u_" + str(where["suggested_id"])
            raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
            answerset = AnswerSet(raw_answer, parser.parse_queryresult)

        output_where[idx]["target_var"] = target_var
        sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query), ds.parser.parse_sparql)
        if (answerset == qapair.answerset) != (sparql == qapair.sparql):
            print("error")

        if answerset == qapair.answerset:
            correct = True
            output_where[idx]["correct"] = True
            output_where[idx]["target_var"] = target_var
        else:
            if target_var == "?u_0":
                target_var = "?u_1"
            else:
                target_var = "?u_0"
            raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
            print("Q_H ",)
            # print(raw_answer)
            print("Q_")
            answerset = AnswerSet(raw_answer, parser.parse_queryresult)

            sparql = SPARQL(kb.sparql_query(where["where"], target_var, count_query, ask_query), ds.parser.parse_sparql)
            if (answerset == qapair.answerset) != (sparql == qapair.sparql):
                print("error")

            if answerset == qapair.answerset:
                correct = True
                output_where[idx]["correct"] = True
                output_where[idx]["target_var"] = target_var

    return "correct" if correct else "-incorrect", output_where


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    utility.setup_logging()

    # with open('data/lcquad20/linked_answer.json', 'r', encoding='utf-8') as f:
    #     linked_answer = json.load(f)
    #     f.close()
    #
    # print(len(linked_answer))
    # print(linked_answer[2])
    # count = 0
    # for index, record in enumerate(linked_answer):
    #     if 'results' in record['answers']:
    #         if len(record['answers']['results']['bindings']) > 0:
    #             print(f"indice: {index}")
    #             print(f"id: {record['id']}")
    #             print(f"question: {record['question']}")
    #             print(f"sparql_query: {record['sparql_query']}")
    #             print(f"answers: {record['answers']}")
    #             print(f"answers-results: {record['answers']['results']}")
    #             print(f"answers-results-bindings: {record['answers']['results']['bindings']}")
    #             # print(record)
    #         count += 1
    #
    # print(f"total questions with answers: {count}")
    # exit()

    ds = LC_Qaud20_Linked(path="data/lcquad20/linked_answer.json")
    ds.load()
    ds.parse()

    if not ds.parser.kb.server_available:
        logger.error("Server is not available. Please check the endpoint at: {}".format(ds.parser.kb.endpoint))
        sys.exit(0)

    parser = LC_Qaud20Parser()
    kb = parser.kb

    stats = Stats()
    linker = GoldLinker()
    output_file = 'lcquad20_gold'

    tmp = []
    output = []
    na_list = []

    for qapair in ds.qapairs:
        print('='*10)
        stats.inc("total")
        output_row = {"question": qapair.question.text,
                      "id": qapair.id,
                      "query": qapair.sparql.query,
                      "answer": "",
                      "features": list(qapair.sparql.query_features()),
                      "generated_queries": []}

        if qapair.answerset is None or len(qapair.answerset) == 0:
            stats.inc("query_no_answer")
            output_row["answer"] = "-no_answer"
            na_list.append(output_row['id'])
        else:
            result, where = qg(linker, ds.parser.kb, ds.parser, qapair, False)
            stats.inc(result)
            output_row["answer"] = result
            newwhere = []
            for iwhere in where:
                if iwhere not in newwhere:
                    newwhere.append(iwhere)
            output_row["generated_queries"] = newwhere
            logger.info(result)

        logger.info(stats)
        output.append(output_row)

        if stats["total"] % 100 == 0:
            with open("output/{}.json".format(output_file), "w") as data_file:
                json.dump(output, data_file, sort_keys=True, indent=4, separators=(',', ': '))

    with open("output/{}.json".format(output_file), "w") as data_file:
        json.dump(output, data_file, sort_keys=True, indent=4, separators=(',', ': '))
    print('stats: ', stats)

    with open('output/na_list_lcquad20_gold.txt', 'w') as f:
        for i in na_list:
            f.write("{}\n".format(i))
