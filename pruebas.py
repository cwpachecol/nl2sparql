from common.container.question import Question
from parsers.lc_quad10 import LC_Qaud10Parser
from parsers.answerparser import AnswerParser

raw_question = "SELECT COUNT(DISTINCT ?uri) WHERE {?uri <http://dbpedia.org/ontology/director> <http://dbpedia.org/resource/Stanley_Kubrick>  . }"
custom_parser = LC_Qaud10Parser()
custom_question = Question(raw_question, custom_parser.parse_question)

print(custom_question.raw_question)
print(custom_question.text)

