import re
import copy
import torch
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
from typing import List, Dict
from .judge import judge_sentence
from .extract import _extract_knowledge
from methods.common.context import has_pronoun
from methods.common.classify import classify_rules
from .self_kg_template_zh import RELATION_TEMPLATE_ZH, RELATION_TOKENS_ZH, CONTRADICT_TEMPLATE_ZH, CONTRADICT_TOKENS_ZH
from .self_kg_template_en import RELATION_TEMPLATE_EN, RELATION_TOKENS_EN, CONTRADICT_TEMPLATE_EN, CONTRADICT_TOKENS_EN
from utils.utils import inverse_softmax, match
from utils.llm import LLM, NLIModel


get_edge_key = lambda nodes: "-".join(nodes)
def remove_duplicate(lis):
    dic = OrderedDict()
    for ele in lis:
        dic[ele] = None
    return list(dic.keys())


class Graph:
    def __init__(self, nodes, language):
        self.nodes = nodes
        self.edges = defaultdict(list)

        self.sample_query_method = 'partial' # How to construct sample queries based on key words
        self.sample_method = 'strict'  # Sampled when exactly-matches(`strict`) or any-word-matches(`relaxed`)

        self.edge_extractor = None  # Extract KG edges with `llm` or None

        self.merge_edges = True  # Merge edges that are identical or contradictary

        self.add_residual = False  # Add verified sentence into KG

        self.verbose = True  # Display self-kg logs

        self.language = language
        self.relation_template = RELATION_TEMPLATE_ZH if language == 'zh' else RELATION_TEMPLATE_EN
        self.contradict_template = CONTRADICT_TEMPLATE_ZH if language == 'zh' else CONTRADICT_TEMPLATE_EN
        self.relation_tokens = RELATION_TOKENS_ZH if language == 'zh' else RELATION_TOKENS_EN
        self.contradict_tokens = CONTRADICT_TOKENS_ZH if language == 'zh' else CONTRADICT_TOKENS_EN

    def get_related_nodes(self, query):
        # since entity in the query might be plural
        return [n for n in self.nodes if n in query.lower()]

    def sample(self, sentence):
        # return remove_duplicate([e for li in self.edges.values() for e in li])
        related_nodes = self.get_related_nodes(sentence)

        def get_sample_queries(nodes):
            queries = []
            if self.sample_query_method == 'individual':
                rg = [1]
            elif self.sample_query_method == 'partial':
                rg = range(1, 3)
            elif self.sample_query_method == 'all':
                rg = [len(nodes)]
            else:
                raise ValueError()

            for m in rg:
                combinations = list(itertools.combinations(nodes, m))
                for combine in combinations:
                    queries.append(get_edge_key(combine))

            return queries

        sampled_edges = []
        for query in get_sample_queries(related_nodes):
            for key in self.edges:
                if self.sample_method == 'strict':
                    if query == key:
                        sampled_edges.extend(self.edges[key])
                elif self.sample_method == 'relaxed':
                    if query in key:
                        sampled_edges.extend(self.edges[key])
                else:
                    raise ValueError()

        sampled_edges = remove_duplicate(sampled_edges)  # remove duplicate
        if self.verbose:
            print(f"[sample] nodes {related_nodes} and edges {sampled_edges}")
        return sampled_edges
    
    def nli(self, sentence, other, model):
        if isinstance(model, LLM):
            def relation_post_process(prob, response, output):
                assert any([w in response.lower() for w in self.relation_tokens])
                return prob, response, output

            prompt = self.relation_template.replace("##SENTENCE_A##", sentence).replace("##SENTENCE_B##", other)
            _, response, _ = model.understand(prompt, post_process=relation_post_process)

        elif isinstance(model, NLIModel):
            _, response, _ = model.understand(sentence, other)

        else:
            raise ValueError()

        return response

    def union_edges(self, edges, other, model, nli_model=None):
        def merge_edge(new_edge, old_edge, model, nli_model):
            nli_model = nli_model or model
            response = self.nli(new_edge, old_edge, nli_model)
            if self.relation_tokens[0] in response.lower():  # identical, keep the original sentence
                merged = [new_edge]
                if self.verbose:
                    print(f"[merge] identical: {new_edge} and {old_edge}")

            elif self.relation_tokens[2] in response.lower(): # compossible, not contradict nor identical
                merged = [new_edge, old_edge]

            elif self.relation_tokens[1] in response.lower(): # contradictory
                try:
                    def contradict_post_process(prob, response, output):
                        assert any(w in response.lower() for w in self.contradict_tokens[0] + self.contradict_tokens[1]) 
                        return prob, response, output

                    prompt = self.contradict_template.replace("##SENTENCE_A##", new_edge).replace("##SENTENCE_B##", old_edge)
                    _, response, _ = model.understand(prompt, post_process=contradict_post_process)
                    if any(w in response.lower() for w in self.contradict_tokens[0]):
                        merged = [new_edge]
                    elif any(w in response.lower() for w in self.contradict_tokens[1]):
                        merged = [old_edge]
                    else:
                        raise ValueError()
                except:
                    merged = [new_edge, old_edge]

                if self.verbose:
                    print(f"[merge] contradictory: {new_edge} and {old_edge}. Keep {merged[0]}")

            else:
                raise ValueError()

            return merged

        for new in other:
            buffer = []
            for item in edges:
                merged = merge_edge(new, item, model, nli_model)
                buffer.extend(merged)
            edges = remove_duplicate(buffer)

        return edges

    def update_kg(self, edges, model, nli_model):
        for key, candidates in edges.items():
            assert isinstance(candidates, list)

            # if edges are new
            if key not in self.edges:
                self.edges[key] = candidates
                if self.verbose:
                    print(f"[merge] edge {key} is new")

            else:
                buffer = copy.deepcopy(self.edges[key])
                if self.merge_edges:
                    buffer = self.union_edges(buffer, candidates, model, nli_model)
                else:
                    buffer.extend(candidates)
                self.edges[key] = remove_duplicate(buffer)


def extract_edges(graph, response, model, language):
    sentences = re.split(r'[.!?]\s+|\n+|。', response)
    sentences = list(filter(lambda s: any(n in s.lower() for n in graph.nodes), sentences))

    if graph.edge_extractor == 'llm':
        _, knowledges, _, _ = _extract_knowledge(sentences, 'global', model, language, ignore_errors=True)
    else:
        knowledges = [sentences]

    phased_knowledges = []
    for line in knowledges:
        for item in line:
            item = item.strip()

            if language == 'zh':
                ignored_words = ['正确', '错误', '无法判断', '因此', '所以', '但是', '然而', '公认', '共识', '注意',
                                 '不同', '具体', '所有', '考虑', '检查']
            else:
                ignored_words = ['true', 'false', 'accurate', 'correct', 'incorrect', 'erroneous', 'not sure',
                                 'therefore', 'so', 'but', 'though', 'although', 'however',
                                 'all', 'consider', 'check', 'notice',
                                 'attention', 'personal', 'answer']

            if has_pronoun(item, language=language):
                continue
            elif any(match(w, item, language) for w in ignored_words):
                continue
            elif classify_rules(item, verbose=False, language=language):
                continue
            phased_knowledges.append(item)

    # get new edges
    edges = defaultdict(list)
    for knowledge in phased_knowledges:
        knowledge = knowledge.replace('（', '(').replace('）', ')')
        knowledge = re.sub(r"\([^\(\)]+\)", "", knowledge)

        # find related nodes
        related_nodes = graph.get_related_nodes(knowledge)
        if related_nodes:
            # record new edges
            key = get_edge_key(related_nodes)
            edges[key].append(knowledge)

    edges = {k: remove_duplicate(v) for k, v in edges.items()}
    if graph.verbose:
        print("[new edges]", edges)

    return edges


def judge_with_self_kg(candidates, entities, model, nli_model, language):
    graph = Graph(nodes=entities, language=language)
    probs = []
    responses = []

    # judge & update kg
    for cid, candidate in enumerate(candidates):
        if candidate:
            if not isinstance(candidate, list):
                candidate = [candidate]

            _probs, _responses = [], []
            for sentence in candidate:
                references = graph.sample(sentence)

                prob, response = judge_sentence(sentence, model, language, context=references)
                _probs.append(torch.from_numpy(prob))
                _responses.append(response)

                new_edges = extract_edges(graph, response, model, language)
                graph.update_kg(new_edges, model, nli_model)

                # add sentence itself into kg
                positive_results = ['a正确', 'a true']
                if graph.add_residual and any(w in response.lower() for w in positive_results): 
                    related_nodes = graph.get_related_nodes(sentence)
                    if related_nodes:
                        key = get_edge_key(related_nodes)
                        graph.update_kg({key: [sentence]}, model, nli_model)

            # aggregate
            _probs = torch.softmax(torch.stack(_probs, dim=0), dim=-1)
            _probs = _probs[:, 1].cpu().numpy()
            prob = np.max(_probs).item()
            response = "\t".join(_responses) + f"\t{_probs}->{prob}"
            prob = inverse_softmax(prob)

        else:
            prob = inverse_softmax(0.)
            response = f"No knowledges"

        probs.append(prob)
        responses.append(response)

    if model.verbose:
        print("graph")
        print(graph.nodes)
        print(graph.edges)

    return probs, responses
