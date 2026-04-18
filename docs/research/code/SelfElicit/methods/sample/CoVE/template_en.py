PLAN_TEMPLATE_EN = """\
Based on the sentence provided, generate a series of verification questions that test the factual claims made in the sentence. \
These questions should help evaluate the accuracy and completeness of the information in the sentence.

Example 1:
Sentence: Angina pectoris can be relieved by sublingual nitroglycerin tablets in emergency situations.
Question:
* What medications or treatments commonly used for relieving angina pectoris in emergency situations?
* Are nitroglycerin tablets taken sublingually (under the tongue) for angina pectori in emergency situations?

Example 2:
Sentence: Each tablet of the thrombolytic and activating tablet weighs 0.76g.
Question:
* What is the weight of each thrombolytic and activating tablet?

Your task:
Sentence: ##SENTENCE##
Question:
"""
PLAN_TEMPLATE_EN_WIKIBIO = """\
Based on the sentence provided, generate a series of verification questions that test the factual claims made in the sentence. \
These questions should help evaluate the accuracy and completeness of the information in the sentence.

Example 1:
Sentence: Gerardo Fernandez Fe was born in Madrid on June 16, 1966.
Question:
* Where was Gerardo Fernandez Fe born?
* When was Gerardo Fernandez Fe born?

Example 2:
Sentence: After the war, Harrer became a professor of geography and later served as an advisor to the Austrian government on mountain-climbing.
Question:
* What kind of professor did Harrer became after the war?
* Which government did Harrer served as an advisor to on mountain-climbing?
* Which field did Harrer served as an advisor to the Austrian government on?

Your task:
Sentence: ##SENTENCE##
Question:
"""

EXECUTE_TEMPLATE_2STEP_EN = """\
You will be provided with multiple questions. Please answer each question, numbering your responses starting from 1. \
Separate each answer with a blank line.

Questions:
##QUESTIONS##

Answers:
"""
EXECUTE_TEMPLATE_FACTOR_EN = """\
Question:
##QUESTION##

Answer:
"""

VERIFY_TEMPLATE_EN = """\
##BLOCKS##

Based on the above question-answer pairs, judge that whether sentence "##SENTENCE##" is correct or incorrect.
"""
VERIFY_BLOCK_EN = """\
Question: ##QUESTION##
Answer: ##ANSWER##
"""
VERIFY_TARGETS_EN = [['correct', 'accurate'], ['incorrect', 'inaccurate', 'erroneous']]
