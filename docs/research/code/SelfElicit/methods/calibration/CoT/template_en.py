TEMPLATE_EN = """\
Sentence: ##QUERY##

Please use chain of thought reasoning to determine whether the sentence is correct or incorrect.
In the end of your reasoning, summary your result and choose your decision from: [A Correct], [B Incorrect] or [C Not sure].
Please analyze:
"""
TARGETS_EN = ['Correct', 'Incorrect', 'Not sure']
