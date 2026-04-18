EXTRACT_CONCEPTS_TEMPLATE_EN = """
Identify named entities in the sentence, including but not limited to: \
symptoms, signs, diseases, tests/examinations, surgical/nursing/treatment procedures, medications, behaviors, substances, \
observational indicators, human structures, numerical values, and time, etc. 
Please ignore the subject of the sentence and only extract other specific information related to the subject. 
Please present each concept in a line starting with * .
##FEWSHOTS##
Your task:
Sentence: ##QUERY##
Concepts:
"""
EXTRACT_CONCEPTS_FEW_SHOTS_EN = [
    {
        "query": "Angina pectoris due to coronary heart disease can be relieved by sublingual nitroglycerin tablets as emergency treatment.",
        "concept": "*sublingual\n*nitroglycerin tablets\n*emergency relief"
    },
    {
        "query": "Benzapril can be taken orally or administered via tubing, with the dosage determined by the physician's prescription.",
        "concept": "*oral\n*administered via tubing"
    },
    {
        "query": "The specification for Xiaoshuantongluo tablets is 0.76g per tablet, taking 3 tablets at a time, 3 times a day.",
        "concept": "*0.76g\n*3 tablets\n*3 times"
    },
    {
        "query": "Myocardial injury markers include troponin I (＜0.03μg/L), myoglobin (male, 28-72ng/mL; female, 25-58ng/mL), and creatine kinase isoenzyme (＜0.05U/L).",
        "concept": "*troponin I\n*0.03μg/L\n*myoglobin\n*male\n*28-72ng/mL\n*female\n*25-58ng/mL\n*creatine kinase isoenzyme\n*0.05U/L"
    },
    {
        "query": "Generally, patients require 2000kcal (4 bottles or 4 bags) of Benzapril per day to meet the body's nutritional needs.",
        "concept": "*2000kcal\n*4 bottles or 4 bagsBenzapril"
    }
]

QUESTION_GENERATION_TEMPLATE_EN = """\
Sentence: ##QUERY##
Create a yes/no question about the entity "##CONCEPT##" from the above sentence.\
The question must include the entity.
Question:
"""
QUESTION_GENERATION_FEW_SHOTS_EN = [
    {
        "query": "Angina pectoris in coronary heart disease can be relieved by sublingual nitroglycerin tablets in emergency situations.",
        "concept": "sublingual administration",
        "question": "Is sublingual administration the method for using nitroglycerin tablets in angina pectoris of coronary heart disease?"
    },
    {
        "query": "Angina pectoris in coronary heart disease can be relieved by sublingual nitroglycerin tablets.",
        "concept": "nitroglycerin tablets",
        "question": "Are nitroglycerin tablets emergency relief medication for angina pectoris in coronary heart disease?"
    },
    {
        "query": "Generally, patients need 2000kcal (4 bottles or 4 bags) of Benazepril daily to meet the body's nutritional requirements.",
        "concept": "2000kcal (4 bottles or 4 bags)",
        "question": "Is 2000kcal the daily requirement of Benazepril for the average patient?"
    },
    {
        "query": "Each tablet of the thrombolytic and activating tablet weighs 0.76g, with a dosage of 3 tablets at a time, three times a day.",
        "concept": "0.76g",
        "question": "Is 0.76g the specification for each tablet of the thrombolytic and activating tablet?"
    }
]

QUESTION_CHECKING_TEMPLATE_EN = """\
Please answer the following question by selecting one of the options: True, False, or Insufficient Information. 
Before making a decision, you must first provide the steps of your reasoning process.
Question: ##QUESTION##
Please reason and select:
"""
QUESTION_CHECKING_TEMPLATE_RETRIEVAL_EN = """\
Please answer the following question by choosing one of the options: Correct, Incorrect, or Insufficient Information, based on the evidence provided. 
Before making a decision, you must first outline the reasoning process steps.
Evidence: ##REFERENCE##
Question: ##QUESTION##
Please reason and select:
"""
QUESTION_CHECKING_TARGETS_EN = ['Correct', 'Incorrect', 'Insufficient Information']
QUESTION_CHECKING_FEW_SHOTS_EN = [
    {
        "reference": "Hypertension is a major public health issue globally, as it increases the risk of heart disease, stroke, and chronic kidney disease. Long-term medication and lifestyle changes, such as diet and exercise, are typically the main methods for controlling hypertension.",
        "question": "Does reducing salt intake help control hypertension?",
        "answer": "Sodium in salt is crucial for maintaining fluid balance in the body and for neuromuscular function, but excessive sodium intake can lead the body to retain more water, which can increase blood volume and thereby raise blood pressure. Reducing sodium intake can help relax blood vessels and lower the pressure within them, resulting in lower blood pressure. Therefore, it can be inferred that reducing salt intake is beneficial for controlling hypertension. Choice: Correct.",
        "answer_retrieval": "The evidence mentions that lifestyle changes are one of the methods for controlling hypertension, and reducing salt intake is usually considered an important part of improving dietary habits. Therefore, it can be inferred that reducing salt intake helps control hypertension. Choice: Correct."
    },
    {
        "reference": "Diabetes is a chronic disease that affects how the body uses blood sugar. Diabetic patients often need to control their blood sugar levels through insulin injections; however, adequate physical activity and proper diet are also crucial for maintaining blood sugar levels.",
        "question": "Do all diabetic patients need to inject insulin?",
        "answer": "Some type 2 diabetic patients have resistance to insulin. Treatment typically starts with lifestyle changes (diet, exercise) and may then require oral medications (such as metformin) to help lower blood sugar levels. Therefore, not all diabetic patients need to inject insulin. Choice: Incorrect.",
        "answer_retrieval": "According to the evidence, while many diabetic patients need to control their blood sugar levels through insulin injections, the importance of adequate physical activity and proper diet is also emphasized. This means not all diabetic patients must rely on insulin injections. Additionally, the type of diabetes (type 1 or type 2) can affect treatment methods. Thus, a general conclusion cannot be made. Choice: Insufficient information."
    },
    {
        "reference": "Heart disease is one of the leading causes of death globally. Effective preventive measures include maintaining a healthy lifestyle, undergoing regular check-ups, and engaging in moderate exercise. Studies show that regular moderate-intensity exercise can significantly reduce the risk of heart disease.",
        "question": "Is high-intensity exercise more effective at preventing heart disease than moderate exercise?",
        "answer": "Excessive high-intensity exercise may put additional stress on the heart, especially for those who are not accustomed to it, at risk for heart disease, or already have heart conditions, and could potentially increase the risk of heart problems. Moreover, insufficient recovery time can lead to overtraining, affecting the immune system, raising the risk of injury, and possibly leading to long-term health issues. Choice: Incorrect.",
        "answer_retrieval": "While the evidence emphasizes the importance of regular moderate exercise in preventing heart disease, it does not specifically compare the efficacy of high-intensity versus moderate exercise. Therefore, based on the current information, we cannot determine if high-intensity exercise is more effective. Choice: Insufficient information."
    },
    {
        "reference": "According to recent clinical studies, antibiotics are very effective in treating certain types of bacterial infections. However, overuse or misuse of antibiotics can exacerbate resistance issues. Doctors recommend using antibiotics only when bacterial infections are confirmed.",
        "question": "Can antibiotics treat all types of infections?",
        "answer": "Antibiotics are medications used against bacterial infections; they can inhibit bacterial growth or kill bacteria. However, antibiotics are ineffective against viral infections such as the common cold, influenza, most sore throats, many ear infections, and many sinus infections. Additionally, antibiotics cannot treat infections caused by fungi or parasites, which require other types of medications. Choice: Incorrect.",
        "answer_retrieval": "Based on the evidence, antibiotics are very effective in treating certain types of bacterial infections. However, this also means that not all types of infections are suitable for antibiotic use, especially since antibiotics are ineffective against viral infections. Therefore, the use of antibiotics is not appropriate for treating all types of infections. Choice: Incorrect."
    }
]
