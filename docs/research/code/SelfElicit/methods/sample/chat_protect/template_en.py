INFORMATION_EXTRACTION_TEMPLATE_EN = """\
You help me identify all the existing entities within it, as well as the relationships between these entities, and output them in a standardized format.
Entities include but are not limited to: symptoms, signs, diseases, examination/test items, surgical/nursing/treatment procedures, drugs, behaviors, substances, \
observation indicators, body structures, numerical values, and time, etc.
Each tuple should consist of a head entity, a relationship, and a tail entity, formatted as (head entity, relationship, tail entity). \
Please use clear delimiters such as parentheses and commas to ensure the tuples are easily identifiable.
If there are multiple tuples, separate each tuple with a line break.
For example:
##FEW_SHOTS##
Now, please provide the tuples based on the following text:
##SENTENCE##
Triples:
"""
INFORMATION_EXTRACTION_TEMPLATE_EN_WIKIBIO = """\
You help me identify all the existing entities within it, as well as the relationships between these entities, and output them in a standardized format.
Entities include but are not limited to: name, place, nationality, event, occupation, award, organization, achievement, date, and time, etc.
Each tuple should consist of a head entity, a relationship, and a tail entity, formatted as (head entity, relationship, tail entity). \
Please use clear delimiters such as parentheses and commas to ensure the tuples are easily identifiable.
If there are multiple tuples, separate each tuple with a line break.
For example:
##FEW_SHOTS##
Now, please provide the tuples based on the following text:
##SENTENCE##
Triples:
"""
INFORMATION_EXTRACTION_FEW_SHOTS_EN = [
    {
        "sentence": "The ApoB/ApoA1 ratio is one of the indicators for assessing coronary heart disease and arteriosclerosis.",
        "tuples": ["(ApoB/ApoA1 ratio, assesses, coronary heart disease)",
                   "(ApoB/ApoA1 ratio, assesses, arteriosclerosis)"]
    },
    {
        "sentence": "Hyperaldosteronism is a disease caused by the excessive secretion of aldosterone from the adrenal cortex.",
        "tuples": ["(Hyperaldosteronism, is, disease)",
                   "(Excessive aldosterone, causes, hyperaldosteronism)",
                   "(Adrenal cortex, secretes, aldosterone)"]
    },
    {
        "sentence": "Ezetimibe is a cholesterol absorption inhibitor that can reduce the absorption of cholesterol in the intestine, thereby lowering blood lipids.",
        "tuples": ["(Ezetimibe, is, cholesterol absorption inhibitor)",
                   "(Ezetimibe, reduces, cholesterol absorption in the intestine)",
                   "(Ezetimibe, lowers, blood lipids)"]
    }
]
INFORMATION_EXTRACTION_FEW_SHOTS_EN_WIKIBIO = [
    {
        "sentence": "Mike Trivisonno joined the 'Monday Night Football' broadcast team for ESPN in 2005.",
        "tuples": ["(Mike Trivisonno, joined, Monday Night Football)",
                   "(Monday Night Football, is, broadcast team)",
                   "(Monday Night Football, belong to, ESPN)"
                   "(Mike Trivisonno, joined in, 2005)"]
    },
    {
        "sentence": "Mike Trivisonno, born on November 27, 1958, is an American former professional football player and television personality.",
        "tuples": ["(Mike Trivisonno, was born on, November 27 1958)",
                   "(Mike Trivisonno, is, American)",
                   "(Mike Trivisonno, is, former professional football player)",
                   "(Mike Trivisonno, is, television personality)", ]
    }
]

STATEMENT_FIRST_TEMPLATE_EN = """\
Please answer the question.
Question: ##QUESTION##
Please generate the first sentence of the answer.

The generated sentence must fill in the blanks in the (##SUBJECT##; ##PREDICATE##; _) triple.
This sentence should contain as little additional information as possible.
"""
STATEMENT_TEMPLATE_EN = """\
You will be given a question and a partially complete answer sentence. \
Please continue to answer that question.
Question: ##QUESTION##
Partial answer sentence: ##STEPS##
Please generate the next sentence of this answer.

The generated sentence must fill in the blanks in the (##SUBJECT##; ##PREDICATE##; _) triple.\
This sentence should contain as little additional information as possible.
"""
STATEMENT_FEW_SHOTS_EN = [
    {
        "question": "What does a ApoB/ApoA1 ratio of 0.54 mean?",
        "steps": "The ApoB/ApoA1 ratio is one of the indicators used to assess coronary heart disease and atherosclerosis.",
        "subject": "ApoB/ApoA1 ratio",
        "predicate": "normal range",
        "statement": "The normal range for the ApoB/ApoA1 ratio is 0.5-1.5."
    },
    {
        "question": "With mild hypertension, low blood potassium, high blood sodium, metabolic alkalosis, decreased plasma renin activity, increased plasma and urine aldosterone, and an increased plasma aldosterone/plasma renin activity ratio, what disease could this be, and how can it be effectively treated to stabilize blood pressure?",
        "steps": "Primary aldosteronism is a disease caused by excessive secretion of aldosterone from the adrenal cortex.",
        "subject": "Aldosterone",
        "predicate": "has an effect",
        "statement": "Aldosterone has the effect of retaining sodium and excreting potassium."
    },
    {
        "question": "Why do I feel uncomfortable in my stomach after taking Ezetimibe?",
        "steps": "Ezetimibe is a cholesterol absorption inhibitor that can reduce cholesterol absorption in the intestine, thereby lowering blood lipids.",
        "subject": "Ezetimibe",
        "predicate": "common side effects",
        "statement": "Common side effects of Ezetimibe include gastrointestinal discomfort, headache, and dizziness."
    }
]
STATEMENT_FEW_SHOTS_EN_WIKIBIO = [
    {
        "question": "Tell me a bio of Chaim Malinowitz.",
        "steps": "Chaim Malinowitz is a rabbi, author, and speaker.",
        "subject": "Chaim Malinowitz",
        "predicate": "is the founder of",
        "statement": "Chaim Malinowitz is the founder of the Jewish Learning Group."
    },
    {
        "question": "Tell me a bio of Jidenna.",
        "steps": "Jidenna Theodore Mobisson (born May 4, 1985) is an American rapper, singer, songwriter, and record producer.",
        "subject": "Jidenna",
        "predicate": "is best known for",
        "statement": "Jidenna is best known for his single \"Classic Man,\" which peaked at number eight on the Billboard Hot R&B/Hip-Hop Songs chart and was nominated for Best Rap/Sung Collaboration at the 58th Grammy Awards."
    }
]

EXPLAIN_FIRST_TEMPLATE_EN = f"""\
I will give you a question, followed by two statements.
Question: ##QUESTION##

Statement 1: ##SENTENCE_A##
Statement 2: ##SENTENCE_B##
Please analyze whether the above two statements contradict each other. Provide only your analysis.
"""
EXPLAIN_TEMPLATE_EN = f"""\
I will give you a question and part of an answer, followed by two statements.
Question: ##QUESTION##
Part of the answer: ##STEPS##

Statement 1: ##SENTENCE_A##
Statement 2: ##SENTENCE_B##

Please analyze whether the above two statements contradict each other. Provide only your analysis.
"""

CONSISTENT_TEMPLATE_EN = """\
Please summarize the above analysis that whether the two statements are contradictory? \
Please respond using ##TARGETS##.
"""
CONSISTENT_TARGET_EN = ['Yes, contradictory', 'No, consistent']
