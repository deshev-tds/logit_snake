import re
import os
import json
from .extract_template_zh import *
from .extract_template_en import *


def _extract_knowledge(sentences, method, model, language,
                       ignore_errors=False):
    core_sentences = []
    core_mapping = []
    for i, sentence in enumerate(sentences):
        if sentence:
            core_sentences.append(re.sub(r"^\d\.", "", sentence.rstrip("。")))
            core_mapping.append(i)

    knowledges = [[]] * len(sentences)
    entities = [[]] * len(sentences)
    cost = []
    if not core_sentences:
        return core_sentences, knowledges, entities, cost

    if method == 'global_json':
        assert language == 'zh'

        prompt = KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_JSON_ZH.replace("##SENTENCE##", "。".join(core_sentences))

        def post_process(prob, response, output):
            response = response.replace('\n', '')
            response = re.findall(r"\[{.*}\]", response)
            if response:
                response = json.loads(response[0])
            else:
                response = []
            return prob, response, output

        _, response, output = model.understand(query=prompt, post_process=post_process)
        cost.append(len(output.scores))
        if response:
            for i in range(min(len(knowledges), len(response))):
                knowledges[i] = [response[i]['claim']]
            knowledges[-1].extend([e['claim'] for e in response[i + 1:]])  # rest

    elif method == 'global_json2':
        assert language == 'zh'

        sentence_str = "<eos>".join(core_sentences) + "<eos>"
        prompt = KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_JSON2_ZH.replace("##SENTENCE##", sentence_str)

        def post_process(prob, response, output):
            response = response.replace('\n', '')
            response = re.findall(r"{.*}", response)[0]
            response = json.loads(response)
            return prob, response, output

        _, response, output = model.understand(query=prompt, post_process=post_process)
        cost.append(len(output.scores))
        for k, knowledge in response.items():
            for sid in range(len(sentences)):  # find sentence id
                if not sentences[sid]:
                    continue
                if k in sentences[sid]:
                    knowledges[sid] = knowledge

    elif method == 'global_noett':
        sentence_str = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(core_sentences)])
        prompt = KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_NOETT_ZH if language == 'zh' else KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_NOETT_EN
        prompt = prompt.replace("##SENTENCE##", "\n" + sentence_str)

        def post_process(prob, response, output):
            if language == 'zh':
                extracted_knowledge = re.findall(r"句子(\d*)中的知识点(.*)", response)
            else:
                extracted_knowledge = re.findall(r"Knowledge points in sentence (\d*)(.*)", response)

            if extracted_knowledge: # retry if not formatted
                pass
            elif len(core_sentences) == 1:
                response_former, response_latter = response.split('知识点') if language == 'zh' else response.split('Knowledge points')
                extracted_knowledge = [(None, response_latter)]
            elif not ignore_errors:
                raise ValueError()
            return prob, (extracted_knowledge), output

        _, (extracted_knowledge), output = model.understand(query=prompt, post_process=post_process)
        cost.append(len(output.scores))

        # extract knowledge
        for k in extracted_knowledge:
            sid, item = k
            if not sid:
                sid = 1
            sid = core_mapping[int(sid) - 1]
            assert sentences[sid], f"Extract knowledge from non-knowledge sentence {sid}."
            item = re.findall(r"\[([^\[\]]*)\]", item)
            knowledges[sid] = item
        entities = []

    elif method == 'global':
        sentence_str = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(core_sentences)])
        prompt = KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_ZH if language == 'zh' else KNOWLEDGE_EXTRACTION_TEMPLATE_GLOBAL_EN
        prompt = prompt.replace("##SENTENCE##", "\n" + sentence_str)

        def post_process(prob, response, output):
            if language == 'zh':
                extracted_entity = re.findall(r"句子(\d*)中的命名实体(.*)", response)
                extracted_knowledge = re.findall(r"句子(\d*)中的知识点(.*)", response)
            else:
                extracted_entity = re.findall(r"Named entities in sentence (\d*)(.*)", response)
                extracted_knowledge = re.findall(r"Knowledge points in sentence (\d*)(.*)", response)

            if extracted_entity and extracted_knowledge: # retry if not formatted
                pass
            elif len(core_sentences) == 1:
                response_former, response_latter = response.split('知识点') if language == 'zh' else response.split('Knowledge points')
                extracted_entity = [(None, response_former)]
                extracted_knowledge = [(None, response_latter)]
            elif not ignore_errors:
                raise ValueError()
            return prob, (extracted_entity, extracted_knowledge), output

        _, (extracted_entity, extracted_knowledge), output = model.understand(query=prompt, post_process=post_process)
        cost.append(len(output.scores))

        # extract entity
        for e in extracted_entity:
            sid, item = e
            if not sid:
                sid = 1
            sid = core_mapping[int(sid) - 1]
            assert sentences[sid], f"Extract entity from non-knowledge sentence {sid}."
            if language == 'zh':
                if '：' in item:
                    item = item[item.index('：') + 1:]
                item = item.replace('。', '').split('、')
            else:
                if ':' in item:
                    item = item[item.index(':') + 1:]
                item = item.split(',')

            entities[sid] = list(filter(lambda e: '无' != e, item))

        # extract knowledge
        for k in extracted_knowledge:
            sid, item = k
            if not sid:
                sid = 1
            sid = core_mapping[int(sid) - 1]
            assert sentences[sid], f"Extract knowledge from non-knowledge sentence {sid}."
            item = re.findall(r"\[([^\[\]]*)\]", item)
            knowledges[sid] = item

    elif method == 'local':
        for sid, sentence in enumerate(sentences):
            if sentence:
                prompt = KNOWLEDGE_EXTRACTION_TEMPLATE_LOCAL_ZH if language == 'zh' else KNOWLEDGE_EXTRACTION_TEMPLATE_LOCAL_EN
                prompt = prompt.replace("##SENTENCE##", sentence)

                _, response, output = model.understand(query=prompt)
                cost.append(len(output.scores))
                knowledge = re.findall(r"\[([^\[\]]*)\]", response)
                knowledges[sid] = knowledge

    knowledges = [list(filter(lambda e: len(e) > 8, k)) for k in knowledges]
    return core_sentences, knowledges, entities, cost


def extract_knowledge(sentences, method, model, sample_id, knowledge_cache, language):
    cache_file = os.path.join(str(knowledge_cache), f"{sample_id}.json")

    if method:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding='utf-8') as fp:
                cache = json.load(fp)
                entities = cache['entities']
                knowledges = cache['knowledges']
                core_sentences = cache['core_sentences']
                for cost in cache['cost']:
                    model.counter.step(cost)

        else:
            core_sentences, knowledges, entities, cost = _extract_knowledge(sentences, method, model, language)

            os.makedirs(knowledge_cache, exist_ok=True)
            with open(cache_file, "w", encoding='utf-8') as fp:
                json.dump({'sentences': sentences,
                           'entities': entities,
                           'core_sentences': core_sentences, 
                           'knowledges': knowledges, 
                           'cost': cost}, fp, ensure_ascii=False)

        # format entities
        phased_entities = []
        for item in entities:
            for e in item:
                e = e.replace('（', '(')
                if '(' in e:
                    e = e[:e.index('(')]
                phased_entities.append(e.strip().lower())
        phased_entities = list(set(phased_entities))
        return knowledges, phased_entities

    else:
        return [[]] * len(sentences), []
