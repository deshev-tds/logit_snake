FACT_EN = """\
Generate some facts about the statement:
##QUERY##
Supporting Facts:
"""

REFLECTION_EN = """\
Given above facts provided, what is your reasoning?
Reasoning:
"""

TEMPLATE_EN = """\
Description: ##QUERY##
Is the above description:
##TARGETS##
Your answer is:
"""
TARGETS_EN = ['A True', 'B False', 'C Not sure']
