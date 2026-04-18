TEMPLATE_EN = """\
# Your Task #
Sentences: 
##QUERY##

Please determine whether each sentence is factually correct or incorrect one by one.
For each sentence, provide your reasoning and choose your decision from: [A Correct], [B Incorrect] or [C Not sure].

# Output Format #
##FORMAT##
"""
TARGETS_EN = ['Correct', 'Incorrect', 'Not sure']
