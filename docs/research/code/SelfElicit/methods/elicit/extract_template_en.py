KNOWLEDGE_EXTRACTION_TEMPLATE_LOCAL_EN = """\
You are a knowledge extractor. I have a sentence, and I need you to identify all the knowledge points from it. Here are some examples:
Example 1:
Sentence: Aldosteronism is a disease caused by the excessive secretion of aldosterone by the adrenal cortex.
Medical knowledge points: [Aldosteronism is a disease][Aldosteronism is caused by excessive aldosterone][Adrenal cortex secretes aldosterone]
Example 2:
Sentence: Ezetimibe can reduce the absorption of cholesterol in the intestine, thereby lowering blood lipids.
Medical knowledge points: [Ezetimibe can reduce the absorption of cholesterol in the intestine][Ezetimibe can lower blood lipids]
Example 3:
Sentence: Side effects after taking Gefitinib, including fever, rash, diarrhea, etc., will disappear on their own within a few weeks after discontinuing the medication.
Medical knowledge points: [Taking Gefitinib may cause side effects][Side effects of Gefitinib include fever][Side effects of Gefitinib include rash][Side effects of Gefitinib include diarrhea][Side effects of Gefitinib disappear on their own within a few weeks after discontinuation]
Your task is to provide the knowledge points based on the following sentence:
Sentence: ##SENTENCE##
Knowledge points:
"""

KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_NOETT_EN = """\
You are a knowledge extractor. Your task is to identify the knowledge points from the sentences.

Steps:
1. For each sentence, extract all the related knowledge points, ensuring the semantic integrity of the points, and that they can be understood independently from the original sentence.
If independent knowledge points cannot be extracted, please return the original sentence directly.
Please use the format “Knowledge points in sentence 1: [Knowledge point 1][Knowledge point 2]” to list all the knowledge points you find.

Example 1:
Sentence:
1. Primary aldosteronism is a disease caused by excessive secretion of aldosterone from the adrenal cortex.
2. Its main clinical symptoms include hypertension, neuromuscular dysfunction, hypokalemic nephropathy, and pyelonephritis.
Knowledge points:
1. Knowledge points in sentence 1: [Primary aldosteronism is a disease][Primary aldosteronism is caused by excessive aldosterone][The adrenal cortex secretes aldosterone]
2. Knowledge points in sentence 2: [Main clinical symptoms of primary aldosteronism include hypertension][Main clinical symptoms of primary aldosteronism include neuromuscular dysfunction][Main clinical symptoms of primary aldosteronism include hypokalemic nephropathy][Main clinical symptoms of primary aldosteronism include pyelonephritis]

Example 2:
Sentence:
1. Ezetimibe can reduce cholesterol absorption in the intestine, thereby lowering blood lipids.
2. Its recommended dosage is once daily, 10mg each time, and it can be taken alone, or in combination with statins, or with fenofibrate.
3. This drug can be taken at any time within a day, on an empty stomach or simultaneously with food.
Knowledge points:
1. Knowledge points in sentence 1: [Ezetimibe can reduce cholesterol absorption][Cholesterol is absorbed in the intestine][Ezetimibe can lower blood lipids]
2. Knowledge points in sentence 2: [The recommended dosage of Ezetimibe is once daily][The recommended dosage of Ezetimibe is 10mg each time][Ezetimibe can be taken alone][Ezetimibe can be used in combination with statins][Ezetimibe can be used in combination with fenofibrate]
3. Knowledge points in sentence 3: [Ezetimibe can be taken at any time within a day][Ezetimibe can be taken on an empty stomach][Ezetimibe can be taken simultaneously with food]

Your task is to provide knowledge points based on the following sentence: 
##SENTENCE##
Knowledge points:
"""

KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_EN = """\
You are a knowledge extractor. Your task is to identify named entities from the given sentences and extract the knowledge points related to these entities.

Steps:
1. For each sentence, identify the named entities within. Named entities include, but are not limited to: symptoms, signs, diseases, examination/test items, surgical/nursing/treatment procedures, drugs, behaviors, substances, observation indicators, human structures, numerical values, and times, etc.
Please use the format “Named entities in sentence 1: Entity 1 (Type 1)” to list all the named entities you find.
2. For each identified named entity, extract all the related knowledge points, ensuring the semantic integrity of the points, and that they can be understood independently from the original sentence.
If independent knowledge points cannot be extracted, please return the original sentence directly.
Please use the format “Knowledge points in sentence 1: [Knowledge point 1][Knowledge point 2]” to list all the knowledge points you find.

Example 1:
Sentence:
1. Primary aldosteronism is a disease caused by excessive secretion of aldosterone from the adrenal cortex.
2. Its main clinical symptoms include hypertension, neuromuscular dysfunction, hypokalemic nephropathy, and pyelonephritis.
Named entities:
1. Named entities in sentence 1: Primary aldosteronism (Disease), adrenal cortex (Human structure), aldosterone (Substance)
2. Named entities in sentence 2: hypertension (Symptom), neuromuscular dysfunction (Symptom), hypokalemic nephropathy (Disease), pyelonephritis (Disease)
Knowledge points:
1. Knowledge points in sentence 1: [Primary aldosteronism is a disease][Primary aldosteronism is caused by excessive aldosterone][The adrenal cortex secretes aldosterone]
2. Knowledge points in sentence 2: [Main clinical symptoms of primary aldosteronism include hypertension][Main clinical symptoms of primary aldosteronism include neuromuscular dysfunction][Main clinical symptoms of primary aldosteronism include hypokalemic nephropathy][Main clinical symptoms of primary aldosteronism include pyelonephritis]

Example 2:
Sentence:
1. Ezetimibe can reduce cholesterol absorption in the intestine, thereby lowering blood lipids.
2. Its recommended dosage is once daily, 10mg each time, and it can be taken alone, or in combination with statins, or with fenofibrate.
3. This drug can be taken at any time within a day, on an empty stomach or simultaneously with food.
Named entities:
1. Named entities in sentence 1: Ezetimibe (Drug), cholesterol (Substance), intestine (Human structure), blood lipids (Substance)
2. Named entities in sentence 2: once daily (Numerical value), 10mg (Numerical value), taken alone (Behavior), statins (Drug), fenofibrate (Drug)
3. Named entities in sentence 3: a day (Time), any time (Time), on an empty stomach (Behavior), food (Substance), taken (Behavior)
Knowledge points:
1. Knowledge points in sentence 1: [Ezetimibe can reduce cholesterol absorption][Cholesterol is absorbed in the intestine][Ezetimibe can lower blood lipids]
2. Knowledge points in sentence 2: [The recommended dosage of Ezetimibe is once daily][The recommended dosage of Ezetimibe is 10mg each time][Ezetimibe can be taken alone][Ezetimibe can be used in combination with statins][Ezetimibe can be used in combination with fenofibrate]
3. Knowledge points in sentence 3: [Ezetimibe can be taken at any time within a day][Ezetimibe can be taken on an empty stomach][Ezetimibe can be taken simultaneously with food]

Your task is to provide named entities and knowledge points based on the following sentence: 
##SENTENCE##
Named entities:
"""