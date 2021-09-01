from parsers.dbnqa_linked import DBNQA_Linked
from parsers.dbnqa import DBNQAParser
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
from config_dbnqa import parse_args


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
    # print("$"*50)
    # print("Entidades:")
    # print(len(entities))
    # print(entities[0])
    # for ent in entities:
    #     print(ent.uris)
    #     for u in ent.uris:
    #         print(u)
    # print("$" * 50)
    double_relation = False
    entities_uris = [eu for eu in qapair.sparql.uris if eu.is_entity()]
    generic_uris = [gu for gu in qapair.sparql.uris if gu.is_generic()]
    no_relation_uris = [nu for nu in qapair.sparql.uris if not nu.is_ontology() and not nu.is_type()]

    relation_uris = [u for u in qapair.sparql.uris if u.is_ontology() or u.is_type()]
    # print("!" * 100)
    # print(len(generic_uris))
    # for gu in generic_uris:
    #     print(gu.raw_uri)
    # print("!" * 100)
    #
    # print("@" * 100)
    # print(len(entities_uris))
    # for eu in entities_uris:
    #     print(eu.raw_uri)
    # print("@" * 100)
    # print("&" * 100)
    # print(len(no_relation_uris))
    # for nru in no_relation_uris:
    #     print(nru.raw_uri)
    # print("&" * 100)
    # print("#" * 100)
    # print(len(relation_uris))
    # for ru in relation_uris:
    #     print(ru.raw_uri)
    # print("#"*100)
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
            print("Q_H ", )
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
    global args
    args = parse_args()

    data_dir = args.data
    linked_answer_json_file = os.path.join(data_dir, 'linked_answer.json')

    logger = logging.getLogger(__name__)
    utility.setup_logging()

    print(linked_answer_json_file)
    ds = DBNQA_Linked(path=linked_answer_json_file)
    ds.load()
    ds.parse()

    # ds.print_pairs(1)

    if not ds.parser.kb.server_available:
        logger.error("Server is not available. Please check the endpoint at: {}".format(ds.parser.kb.endpoint))
        sys.exit(0)

    parser = DBNQAParser()
    kb = parser.kb
    # print(kb.endpoint)
    # print(kb.default_graph_uri)
    # print(kb.sparql_query)
    # print(kb.type_uri)

    stats = Stats()
    linker = GoldLinker()
    # print(linker)

    # output_file = 'dbnqa_gold.json'

    tmp = []
    output = []
    na_list = []

    # //////////////////////////////////////
    output_dir = args.output
    start_dbnqa_gold_element = 0
    dbnqa_gold_list = []
    dbnqa_gold_json_file = os.path.join(output_dir, 'dbnqa_gold.json')
    start_no_dbnqa_gold_element = 0
    na_dbnqa_gold_list = []
    na_dbnqa_gold_txt_file = os.path.join(output_dir, 'na_dbnqa_gold.json')
    # linked_answer_json_file = os.path.join(data_dir, 'linked_answer.json')

    if os.path.isfile(dbnqa_gold_json_file):
        print("existe dbnqa_gold_json_file")
        with open(dbnqa_gold_json_file, 'r', encoding='utf-8') as dbnqag_json_file:
            dbnqa_gold_list = json.load(dbnqag_json_file)
        dbnqag_json_file.close()

        if len(dbnqa_gold_list) > 0 and dbnqa_gold_list[-1].get('id') is not None:
            start_element = int(dbnqa_gold_list[-1].get('id'))

    if os.path.isfile(na_dbnqa_gold_txt_file):
        print("existe na_dbnqa_gold_json_file")
        with open(na_dbnqa_gold_txt_file, 'r', encoding='utf-8') as na_dbnqag_txt_file:
            na_dbnqa_gold_list = na_dbnqag_txt_file.read().split('\n')
        na_dbnqag_txt_file.close()

    # /////////////////////////////////////////

    for qapair in ds.qapairs[start_dbnqa_gold_element:]:
        # print('='*10)
        stats.inc("total")
        output_row = {"question": qapair.question.text,
                      "id": qapair.id,
                      "query": qapair.sparql.query,
                      "answer": "",
                      "features": list(qapair.sparql.query_features()),
                      "generated_queries": []}
        # print(qapair.sparql.query_features())
        # print("="*100)
        if qapair.answerset is None or len(qapair.answerset) == 0:
            stats.inc("query_no_answer")
            output_row["answer"] = "-no_answer"
            na_dbnqa_gold_list.append(output_row['id'])
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
        dbnqa_gold_list.append(output_row)
        print(f'Id: {qapair.id}')
        if int(qapair.id) % 100 == 0:
            with open(dbnqa_gold_json_file, "w") as data_file:
                json.dump(dbnqa_gold_list, data_file, sort_keys=True, indent=4, separators=(',', ': '))
            data_file.close()
            with open(na_dbnqa_gold_txt_file, 'w') as na_data_file:
                for i in na_dbnqa_gold_list:
                    na_data_file.write("{}\n".format(i))
            na_data_file.close()

    with open(dbnqa_gold_json_file, "w") as data_file:
        json.dump(dbnqa_gold_list, data_file, sort_keys=True, indent=4, separators=(',', ': '))
    data_file.close()

    print('stats: ', stats)

    # Questions without answers
    with open(na_dbnqa_gold_txt_file, 'w') as na_data_file:
        for i in na_dbnqa_gold_list:
            na_data_file.write("{}\n".format(i))
    na_data_file.close()
