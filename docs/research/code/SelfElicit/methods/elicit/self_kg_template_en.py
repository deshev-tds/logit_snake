RELATION_TEMPLATE_EN = """\
Please determine the semantic relationship between the following two sentences. There are three possible types of relationships:
1. [entail]: The content of the two sentences is the identical, describing the same aspect of the same object, with consistent content.
2. [contradict]: The two sentences describe the same aspect of the same object, but the content is directly opposite, presenting a contradiction.
3. [neutral]: The two sentences describe different objects, or different aspects of the same object, and can coexist.
Please analyze sentence A and sentence B, and choose one of the relationships. Please briefly explain your reasoning.
Sentence A: ##SENTENCE_A##
Sentence B: ##SENTENCE_B##
Judgment result:
"""
RELATION_TOKENS_EN = ["entail", "contradict", "neutral"]

CONTRADICT_TEMPLATE_EN = """\
Please read the following two sentences.
These two sentences describe the same aspect of the same object, but their content is contradictory. Your task is to judge which sentence is more accurate based on your own understanding.
Sentence A: ##SENTENCE_A##
Sentence B: ##SENTENCE_B##
Judging criteria:
Please consider the logic and factual basis of the sentences. Choose the sentence you think is correct and select from the following two options:
[Sentence A is correct]
[Sentence B is correct]
"""
CONTRADICT_TOKENS_EN = [["sentence a is correct", "sentence a is the correct", "sentence a is more accurate", "[sentence a] is correct"],\
                        ["sentence b is correct", "sentence b is the correct", "sentence b is more accurate", "[sentence b] is correct"]]
