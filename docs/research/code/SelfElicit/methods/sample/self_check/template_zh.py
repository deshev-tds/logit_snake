TARGET_EXTRACTION_TEMPLATE_ZH = """\
下面是针对一个医疗问题的回答步骤。
问题：##QUESTION##
回答步骤：
##STEPS##
其中步骤“##STEP##”的主题是什么？请用一句话给出简短的答案，不要复制步骤。
"""

INFORMATION_COLLECTION_FIRST_TEMPLATE_ZH = """\
医疗问题：##QUESTION##
以下是回答该问题的第一个步骤：
##STEP##
第一个回答步骤包含哪些医疗信息？
"""
INFORMATION_COLLECTION_TEMPLATE_ZH = """\
医疗问题：##QUESTION##
以下是回答该问题的前几个步骤：
##STEPS##

下一个回答步骤##STEP##包含哪些医疗信息？
"""

RERUNNING_FIRST_TEMPLATE_ZH = """\
我们正在回答一个医疗问题，回答中会包含多个步骤。
第一个回答步骤用到的信息有：
##INFORMATION##
第一个回答步骤的主题是：##TARGET##

请根据以上信息生成第一个回答步骤：
[步骤1]
"""
RERUNNING_TEMPLATE_ZH = """\
我们正在回答一个医疗问题，回答中会包含多个步骤。
当前步骤之前的回答步骤包括：
##STEPS##

当前回答步骤包含的信息有：
##INFORMATION##
当前回答步骤的目标是：##TARGET##

请根据以上信息生成当前回答步骤：
[步骤##STEPID##]
"""

COMPARE_TEMPLATE_ZH = """\
以下是医疗问题的2个部分回答：
回答1：##ANSWER_ORIG##
回答2：##ANSWER_RERUN##
逐步比较两种回答内容的关键点，然后检查回答1与回答2的关系是“支持”、“矛盾”还是“无关”。
"""

COMPARE_TARGETS_ZH = ['支持', '矛盾', '无关']
