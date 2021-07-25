from common.container.answer import Answer
from common.container.uri import Uri
from kb.kb import KB
from common.container.question import Question
from parsers.lc_quad10 import LC_Quad10Parser
from parsers.answerparser import AnswerParser
respuesta_en_bruto = " "
tipo_respuesta = "bool"
# tipo_respuesta = "uri"
interprete = LC_Quad10Parser()
ENDPOINT = "http://dbpedia.org/sparql"
GRAPH = "http://dbpedia.org"
kb_ref = KB(ENDPOINT, GRAPH)
question_custom = "SELECT DISTINCT ?uri WHERE {?uri <http://dbpedia.org/property/music> <http://dbpedia.org/resource/Elliot_Goldenthal> . ?uri <http://dbpedia.org/property/screenplay> <http://dbpedia.org/resource/Akiva_Goldsman>  . ?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
print(kb_ref.query(question_custom))
print(kb_ref.check_server())


print(kb_ref.prefix())
# print(kb_ref.parse_uri())
print(kb_ref.shorten_prefix())

# def __parse_answer(answer_type, raw_answer):
#     prefix = kb_ref.prefix()
#     print(prefix)
#     if len(prefix) > 0 and raw_answer.startswith(prefix):
#         raw_answer = kb_ref.shorten_prefix() + raw_answer[len(prefix):]
#     return answer_type, Uri(raw_answer, kb_ref.parse_uri)
#
# respuesta = Answer(tipo_respuesta, respuesta_en_bruto, __parse_answer)
#
# print(respuesta)
#

raw_question = "SELECT COUNT(DISTINCT ?uri) WHERE {?uri <http://dbpedia.org/ontology/director> <http://dbpedia.org/resource/Stanley_Kubrick>  . }"
custom_parser = LC_Quad10Parser()
custom_question = Question(raw_question, custom_parser.parse_question)

custom_parser.
print(custom_question.raw_question)
print(custom_question.text)

