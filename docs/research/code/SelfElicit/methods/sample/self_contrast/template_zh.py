REFLECTION_TEMPLATES0 = """\
请仔细检查以下回答是否包含事实性错误，并提供详细的反馈。
问题：##QUESTION##
回答：##ANSWER##
回答是否包含事实性错误：
"""
REFLECTION_TEMPLATES1 = """\
问题：##QUESTION##
回答：##ANSWER##
请你检查上述回答中是否包含任何的医学专业知识的错误，并提供详细的反馈。
"""
REFLECTION_TEMPLATES2 = """\
问题：##QUESTION##
回答：##ANSWER##
你认为前面的回答正确与否。如果其中存在错误，请指出哪里出了问题。
"""
REFLECTION_TEMPLATES3 = """\
问题：##QUESTION##
回答：##ANSWER##
请仔细评估上述回答的质量，并指出您是否觉得不合适。
"""
REFLECTION_TEMPLATES4 = """\
问题：##QUESTION##
回答：##ANSWER##
上述的回答可能存在一些错误，因此请仔细检查并找出错误。如果你认为根本没有错误，请回答“完全正确”。
"""
REFLECTION_TEMPLATES5 = """\
问题：##QUESTION##
回答：##ANSWER##
请你考虑以上回答中是否能够解决提出的问题。如果回答无法解决问题，请指出。
"""
REFLECTION_TEMPLATES6 = """\
给你一个问题和一个回答，请你帮我检查这个回答能否是对这个问题的直接回应。
问题：##QUESTION##
回答：##ANSWER##
检查：
"""
REFLECTION_TEMPLATES7 = """\
给你一个问题和一个回答，请你帮我检查回答中是否存在专业医疗知识的错误。
问题：##QUESTION##
回答：##ANSWER##
检查错误：
"""
REFLECTION_TEMPLATES_ZH = [REFLECTION_TEMPLATES0, REFLECTION_TEMPLATES1,
                           REFLECTION_TEMPLATES2, REFLECTION_TEMPLATES3,
                           REFLECTION_TEMPLATES4, REFLECTION_TEMPLATES5,
                           REFLECTION_TEMPLATES6, REFLECTION_TEMPLATES7]

CONTRASTING_TEMPLATE_ZH = """\
我想要评估一个医疗问答是否正确。以下是一些评估结果，请你应该仔细比较这些评估结果之间的差异。\
比较时，请你考虑以下问题：
1. 两个评估结果的最终结果是否不同？
2. 它们评估的原因的差异在哪里？
3. 为什么两次评估的结果不同？
对比后，请你根据这些差异生成一个清单。请你应该仔细考虑每个差异及其背后的原因，将它们总结为清单中的一些检查说明。\
这个清单的内容包括：为了对医疗问答进行有效的正确性评估，需要特别注意的地方。

医疗问答正确性的评估结果：
##REFLECTIONS##

输出格式：
##CONTRASTS##
检查清单：{项目1；项目2；...}
"""
CONTRASTING_OUTPUT_TEMPLATE_ZH = """\
评估##REFLECTION_A##和评估##REFLECTION_B##：{差异}\
"""

SUMMARY_TEMPLATE_ZH = """\
给定一个医疗问答、针对这个医疗问答正确性的评估结果、评估结果之间的差异以及一个检查清单。\
请你修改评估结果，消除评估结果之间的差异，并总结最终的评估结果。
步骤：
1. 请根据清单上的要求仔细检查。它可以帮助你解决不同评估结果之间的冲突。
2. 当你修改完评估结果时，请尽可能确保所有修改后的评估结果都一致。\
如果没有，请再次修改，直到所有不一致之处都消除，并且所有评估结果都一致。\

医疗问答：
问：##QUESTION##
答：##ANSWER##

针对上述问答正确性的评估结果：
##REFLECTIONS##

差异：
##DIFFERENCES##

检查清单：
##CHECKLIST##

请根据检查清单，消除不同评估结果之间的差异。
使用“修改后的评估1”句式分别给出评估1、评估2和评估3修改后的结果。
"""

SUMMARY_FORMALIZATION_TEMPLATE_ZH = """\
医疗问答：
问：##QUESTION##
答：##ANSWER##

评估结果：##SUMMARY##

请你帮我总结，评估结果认为上述医疗问答是正确的、错误的还是中立：
"""
SUMMARY_FORMALIZATION_TARGETS_ZH = ["正确", "错误", "中立"]
