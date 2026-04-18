REFLECTION_TEMPLATES0 = """\
Please carefully check whether the following answer contains factual errors and provide detailed feedback.
Question: ##QUESTION##
Answer: ##ANSWER##
Does the answer contain factual errors:
"""
REFLECTION_TEMPLATES1 = """\
Question: ##QUESTION##
Answer: ##ANSWER##
Please check the answer above for any errors in medical expertise and provide detailed feedback.
"""
REFLECTION_TEMPLATES2 = """\
Question: ##QUESTION##
Answer: ##ANSWER##
Do you think the previous answer is correct or not? If there are any errors, please indicate where the problem lies.
"""
REFLECTION_TEMPLATES3 = """\
Question: ##QUESTION##
Answer: ##ANSWER##
Please carefully evaluate the quality of the above answer and indicate whether you find it inappropriate.
"""
REFLECTION_TEMPLATES4 = """\
Question: ##QUESTION##
Answer: ##ANSWER##
The answer above may contain some errors, so please check carefully and identify any mistakes. If you believe there are no errors at all, please respond with "Completely correct."
"""
REFLECTION_TEMPLATES5 = """\
Question: ##QUESTION##
Answer: ##ANSWER##
Please consider whether the answer provided can solve the problem posed. If the answer cannot solve the problem, please point it out.
"""
REFLECTION_TEMPLATES6 = """\
Give you a question and an answer. Please help me check if this answer can be a direct response to this question.  
Question: ##QUESTION##  
Answer: ##ANSWER##  
Check:  
"""
REFLECTION_TEMPLATES7 = """\
Give you a question and an answer. Please help me check if there are any errors in the specialized medical knowledge in the answer.
Question: ##QUESTION##
Answer: ##ANSWER##
Check for errors:
"""
REFLECTION_TEMPLATES_EN = [REFLECTION_TEMPLATES0, REFLECTION_TEMPLATES1,
                           REFLECTION_TEMPLATES2, REFLECTION_TEMPLATES3,
                           REFLECTION_TEMPLATES4, REFLECTION_TEMPLATES5,
                           REFLECTION_TEMPLATES6, REFLECTION_TEMPLATES7]

CONTRASTING_TEMPLATE_EN = """\
I want to evaluate whether a medical Q&A is correct. Below are some evaluation results; please carefully compare the differences between these results.\
When comparing, please consider the following questions:
1. Are the final results of the two evaluations different?
2. Where are the differences in the reasons for their evaluations?
3. Why do the results of the two evaluations differ?
After comparing, please generate a checklist based on these differences. Please carefully consider each difference and the reasons behind them, summarizing them into some checks in the checklist.\
The content of this checklist includes: points that need special attention for effective correctness evaluation of medical Q&A.

Evaluation results of medical Q&A correctness:
##REFLECTIONS##

Output format:
##CONTRASTS##
Checklist: {Item1; Item2, ...}
"""
CONTRASTING_OUTPUT_TEMPLATE_EN = """\
Evaluation ##REFLECTION_A## and Evaluation ##REFLECTION_B##: {Difference}\
"""

SUMMARY_TEMPLATE_EN = """\
Given a medical question and answer, an evaluation of the correctness of this medical question and answer, the differences between the evaluation results, and a checklist.\
Please modify the evaluation results to eliminate the differences between them and summarize the final evaluation results.

Steps:
1. Please carefully check according to the requirements on the checklist. It can help you resolve conflicts between different evaluation results.
2. When you have modified the evaluation results, please ensure that all modified results are as consistent as possible.\
If there are inconsistencies, please modify again until all discrepancies have been eliminated and all evaluation results are consistent. 

Medical Q&A:
Question: ##QUESTION##
Answer: ##ANSWER##
Evaluation results of the correctness of the aforementioned Q&A:
##REFLECTIONS##
Differences:
##DIFFERENCES##
Checklist:
##CHECKLIST##
Please use the checklist to eliminate the differences between the evaluation results.
Use the phrase "Modified Evaluation 1" to present the modified results for Evaluations 1, 2, and 3 respectively.
"""

SUMMARY_FORMALIZATION_TEMPLATE_EN = """\
Sentence: ##SUMMARY##
Please use "positive", "negative" or "neutral" to summarize the attitude of the above sentence.
"""
SUMMARY_FORMALIZATION_TARGETS_EN = ["positive", "negative", "neutral"]
