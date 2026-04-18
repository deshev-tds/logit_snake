KNOWLEDGE_PROMPTING_EN = """\
Generate some knowledge about the statement:
##QUERY##
Knowledge:
"""

TEMPLATE_EN = """\
Description: ##QUERY##
Is the above description:
##TARGETS##
Choose your option from A, B and C. Nothing else:
Your answer is:
"""
TARGETS_EN = ['A True', 'B False', 'C Not sure']
