CLASSIFICATION_TEMPLATE_EN = """\
You will be handling questions and answers related to medical consultations and healthcare. Your task is to categorize a sentence from the response based on its content. Classify the sentence accurately under one of the following categories:
1. [Medical Knowledge]: Includes objective descriptions of medical knowledge, detailing specific diseases, symptoms, medications, methods, etc. Examples include:
  a. Ezetimibe is a cholesterol absorption inhibitor that reduces cholesterol absorption in the gut, thereby lowering blood lipids.
  b. Common pathological classifications for lung cancer include non-small cell lung cancer and small cell lung cancer.
  c. Upper gastrointestinal radiography or endoscopy is used to observe the extent, severity, and recurrence or metastasis of esophageal lesions.
  d. Skin symptoms caused by radiotherapy usually resolve spontaneously within a few weeks after treatment ends.
  e. For patients with coronary heart disease, Isosorbide mononitrate tablets may cause some side effects such as headache and neck pain.
  f. For some low-risk patients, such as those with smaller tumors, better differentiation, and no high-risk factors, observation and follow-up can be chosen instead of immediate adjuvant chemotherapy.
2. [Personal Condition]: Describes the current state of a specific patient (complaints, history, laboratory data, signs), without including treatment or advice. Examples include:
  a. Age 48, tumor marker carcinoembryonic antigen 100.
  b. Lymph node metastasis in the neck after esophageal cancer surgery.
  c. Blood test results show a hemoglobin level of 8.5 g/dL.
3. [Lifestyle]: Discusses health and lifestyle habits other than treatment. Examples include:
  a. Increasing physical exercise can effectively reduce the risk of cardiovascular disease.
  b. Maintaining regular sleep schedules helps improve sleep quality.
  c. Healthy eating habits are crucial for maintaining weight and overall health.
4. [Other]: Sentences that do not fit into any of the above categories, such as emotional expression type, subjective evaluation type, non-medical type, etc.
Please identify which category the following sentence from the response belongs to:
##SENTENCE##
"""
CLASSIFICATION_TARGETS_EN = ['Medical Knowledge', 'Personal Condition', 'Summarization', 'Hospitalization', 'Lifestyle', 'Other', 'Common Sense']
