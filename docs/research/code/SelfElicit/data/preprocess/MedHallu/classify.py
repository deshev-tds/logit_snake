import os
import re
import json
import argparse
from tqdm import tqdm
from data.dataloader import load_dataset
from utils.llm import load_model


CLASSIFICATION_TEMPLATE = """\
你是一个句子分类器。给定一个医疗相关的句子，根据如下类别对其进行分类：
1. [医学知识型]：直接包含或者间接涉及专业医学知识，详细描述具体疾病、症状、指标、药物、方法等。例如：
  a. 依折麦布是一种胆固醇吸收抑制剂，可以减少胆固醇在肠道的吸收，从而降低血脂。
  b. 肺癌常用的病理学分类包括非小细胞肺癌和小细胞肺癌两大类。
  c. 上消化道造影或胃镜检查用于观察食管病变范围、程度以及有无复发或转移。
  d. 放疗引起的皮肤症状通常在放疗结束后几周内自行消退。
  e. 您的血压指标高于正常范围。
  f. 对于一些低风险的患者，例如肿瘤较小、分化较好、没有高危因素等，可以选择观察随访，而不是立即进行辅助化疗。
2. [个人情况型]：客观陈述患者个体的主诉、病史或具体实验室数据、体征等。这是患者当前的状态，不涉及客观的医疗知识。例如：
  a. 患者48岁，肿瘤标志物癌胚抗原100。
  b. 食管癌术后，检查显示颈部淋巴结转移。
  c. 血液检查结果显示当前您的血红蛋白水平为8.5 g/dL。
3. [综述概述型]：只说明有“多种”、“几种”、但未提及具体症状、药物或方法。例如：
  a. 通过多种方法可以缓解食道肿大造成的吞咽困难。
  b. 肝癌晚期患者如果不能进行手术治疗，通常有几种治疗方法。
  c. 上段食管癌可以采用多种治疗方法进行综合治疗。
4. [个性化就医型]：建议患者或个人“就医”，强调治疗方案和诊断需要根据患者的“具体情况”进行“个性化”制定，但不详细描述具体方法或药物。例如：
  a. 建议您立即就医，进行全面的评估和必要的影像学检查。
  b. 具体的治疗方案应由医生根据患者的具体情况制定。
  c. 确诊是否为复发以及病情的具体情况需要结合其他检查结果进行综合评估。
  d. 医生将结合病史、体格检查和其他相关检查结果来确定淋巴结肿大的具体原因，并制定相应的治疗计划。
  e. 如果肿瘤分化程度较高，是否进行辅助化疗需要根据患者的具体情况来决定。
5. [健康生活型]：讨论了除了治疗以外的护理、康复和生活习惯。例如：
  a. 通过增加体育锻炼可以有效降低患心血管疾病的风险。
  b. 坚持规律的作息时间有助于提高睡眠质量。
  c. 健康的饮食习惯对维持体重和整体健康非常重要。
  d. 建议适当运动，保持低盐低脂饮食，多吃高蛋白、富含维生素食物，有助于患者恢复。
6. [其他]：不符合以上任意类别的句子，例如情感表达型、主观评价型、非医学型等。

请分析以下句子所描述的内容，判断其属于哪一种类别，最后使用[类别]给出你的判断。
##SENTENCE##
"""
CLASSIFICATION_TARGETS = ['[医学知识型]', '[个人情况型]', '[综述概述型]', '[个性化就医型]', '[健康生活型]', '[其他]']


def _classify_sentence(sentence, model):
    prompt = CLASSIFICATION_TEMPLATE.replace("##SENTENCE##", sentence)

    _, response, _ = model.understand(query=prompt)

    pred_cls = len(CLASSIFICATION_TARGETS) - 1
    for i, target in enumerate(CLASSIFICATION_TARGETS):
        if target in response:
            pred_cls = i
            break

    return pred_cls, response


def classify_sentence_with_rules(sentences):
    pred_clses, responses = [], []
    for sentence in sentences:
        if any([re.findall(p, sentence) for p in [r"几种", r'以下', r'如下', r'下列', r'以上[^，]', r'参考自【.*】']]):
            pred_cls = 2
            response = f"Key word classify as 综述概述型"

        elif any([re.findall(p, sentence) for p in [r'医生', r'医师', r'医嘱', r'就医', r'家人', r"综合考虑", r"综合评[估价]", r"个性化"]]):
            pred_cls = 3
            response = f"Key word classify as 就医型"

        elif any([re.findall(p, sentence) for p in [r"高蛋白", r"锻炼", r'心态', r'心理', r'心情', r"定期"]]):
            pred_cls = 4
            response = f"Key word classify as 健康生活型"

        else:
            pred_cls, response = _classify_sentence(sentence, model)

        pred_clses.append(pred_cls)
        responses.append(response)

    return pred_clses, responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'val'])

    parser.add_argument("--do_sample", action='store_true')

    parser.add_argument("--data_path", type=str, default="./data/dataset/MedHallu")
    parser.add_argument("--save_path", type=str, default="./cache/classify/")

    args = parser.parse_args()
    args.max_new_tokens = 8192
    args.top_k = args.top_p = args.repetition_penalty = None
    args.model_path = args.model_path.rstrip('/')
    args.model_name = args.model_path.split('/')[-1].lower().replace('-', '_')

    model = load_model(args.__dict__)
    datasets = load_dataset(args.data_path)

    path = os.path.join(args.save_path, args.model_name, args.mode)
    os.makedirs(path, exist_ok=True)

    for i in tqdm(range(len(datasets[args.mode]))):
        sample = datasets[args.mode][i]
        cache_file = os.path.join(path, f"{i}.json")

        if not os.path.exists(cache_file):
            clses, responses = classify_sentence_with_rules(sample['sentences'])

            with open(cache_file, "w", encoding='utf-8') as f:
                json.dump({"sentences": sample['sentences'],
                            "clses": clses,
                            "responses": responses},
                            f, ensure_ascii=False)
