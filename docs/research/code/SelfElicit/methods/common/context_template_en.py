PRONOUN_REPLACEMENT_TEMPLATE_EN = """\
I have a sentence and a context. Please refer to the context, identify the pronouns or unclear references in the sentence. \
Then, through reasoning, determine their specific referents and replace the pronouns with their specific referents.\
For example:

Example 1:
Context: Nivolumab is a drug. Nivolumab is a PD-1 inhibitor. It is commonly used to treat small cell lung cancer. Its common side effects include fatigue, loss of appetite, nausea, etc.
Sentence: It is commonly used to treat small cell lung cancer.
Judgment: The "it" in the sentence refers to "nivolumab," and the updated sentence is "Nivolumab is commonly used to treat small cell lung cancer."

Example 2:
Context: Drug treatment for coronary heart disease patients does not require changing the medication every two years. The main drugs used for treating coronary heart disease include nitrate vasodilators, statins for lowering blood lipids, and aspirin anticoagulants. These drugs can be taken long-term and can help control the condition of patients with coronary heart disease. If the condition is well controlled, there is no need to change the medication every two years.
Sentence: These drugs can be taken long-term and can help control the condition of patients with coronary heart disease.
Judgment: The "these" in the sentence refers to "nitrate vasodilators, statins for lowering blood lipids, and aspirin anticoagulants." The updated sentence is "Nitrate vasodilators, statins for lowering blood lipids, and aspirin anticoagulants can be taken long-term and can help control the condition of patients with coronary heart disease."

Your task is to refer to the context, identify the pronouns or unclear references in the following sentence, and then replace them with specific referents.
##CONTEXT##
##SENTENCE##
Judgment:
"""
