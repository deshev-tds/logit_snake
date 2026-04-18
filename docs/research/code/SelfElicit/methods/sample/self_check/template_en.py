TARGET_EXTRACTION_TEMPLATE_EN = """\
Here are the steps for answering a medical question.
Question: ##QUESTION##
Answering Steps:
##STEPS##
What is the theme of the step "##STEP##"? Please provide a brief answer in one sentence without copying the step.
"""

INFORMATION_COLLECTION_FIRST_TEMPLATE_EN = """\
Medical Question: ##QUESTION##
The first step in answering this question is as follows:
##STEP##
What medical information does the first step include?
"""
INFORMATION_COLLECTION_TEMPLATE_EN = """\
Medical Question: ##QUESTION##
Here are the first few steps to answer the question:
##STEPS##
What medical information is included in the next response step ##STEP##?
"""

RERUNNING_FIRST_TEMPLATE_EN = """\
We are answering a medical question, and the response will involve multiple steps. 
Here are some information: 
##INFORMATION## 
The theme of the first response step is: ##TARGET##

Please generate the first response sentence based on the above information: 
[Step 1]
"""
RERUNNING_TEMPLATE_EN = """\
We are answering a medical question, and the response will include multiple steps. 
The previous response steps include: 
##STEPS##

Here are some information: 
##INFORMATION##

The theme of the current response step is: ##TARGET##

Please generate the current response sentence based on the above information:
[Step ##STEPID##]
"""

COMPARE_TEMPLATE_EN = """\
The following are two sentences answering a medical question:
Answer 1: ##ANSWER_ORIG##
Answer 2: ##ANSWER_RERUN##
Compare the key points from both answers step by step and \
then summarize your answer whether Answer 1 "supports", "contradicts" or "is not directly related to" \
the conclusion in Answer 2. \
Pay special attention to difference in facts.
"""

COMPARE_TARGETS_EN = ['support', 'contradict', 'not directly related']
