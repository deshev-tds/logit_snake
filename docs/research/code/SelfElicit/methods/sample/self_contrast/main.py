import re
from evaluate import Evaluator
from methods import AbstractMethod
from utils.parser import parser, parse_args
from utils.utils import init_env, save_metrics, inverse_softmax
from template_zh import *
from template_en import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class SelfContrast(AbstractMethod):
    """
    Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives
    """

    def __init__(self, args):
        super().__init__(args)

        self.refection_template = REFLECTION_TEMPLATES_ZH if args.language == 'zh' else REFLECTION_TEMPLATES_EN

        self.contrasting_template = CONTRASTING_TEMPLATE_ZH if args.language == 'zh' else CONTRASTING_TEMPLATE_EN
        self.contrasting_output_template = CONTRASTING_OUTPUT_TEMPLATE_ZH if args.language == 'zh' else CONTRASTING_OUTPUT_TEMPLATE_EN

        self.summary_template = SUMMARY_TEMPLATE_ZH if args.language == 'zh' else SUMMARY_TEMPLATE_EN

        self.summary_formalization_template = SUMMARY_FORMALIZATION_TEMPLATE_ZH if args.language == 'zh' else SUMMARY_FORMALIZATION_TEMPLATE_EN
        self.summary_formalization_targets = SUMMARY_FORMALIZATION_TARGETS_ZH if args.language == 'zh' else SUMMARY_FORMALIZATION_TARGETS_EN

    def refection(self, question, answer, model):
        reflections, embeddings = [], []
        for refection_template in self.refection_template:
            refection_template = refection_template.replace("##QUESTION##", question)
            refection_template = refection_template.replace("##ANSWER##", answer)

            _, response, output = model.understand(refection_template, output_hidden_states=True)
            reflections.append(response.replace("\n", ""))
            embedding = output.hidden_states[-1][-1][0, 0]  # last token, last layer, first sample
            assert embedding.ndim == 1
            embeddings.append(embedding.cpu().numpy())

        return reflections, embeddings

    def clustering(self, embeddings):
        embeddings = np.stack(embeddings)
        kmeans = KMeans(n_clusters=self.args.k, random_state=0)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        indices = []
        for i in range(len(centers)):
            cluster_samples_indices = np.where(labels == i)[0]
            distances = np.linalg.norm(embeddings[cluster_samples_indices] - centers[i], axis=1)
            closest_index = cluster_samples_indices[np.argmin(distances)]
            indices.append(closest_index)

        return indices

    def contrasting(self, reflections, model):
        reflections_str = []
        for ref_i, ref in enumerate(reflections):
            if self.args.language == 'zh':
                reflections_str.append(f"评估{ref_i + 1}：{ref}")
            else:
                reflections_str.append(f"Evaluation {ref_i + 1}: {ref}")

        contrasting_str = []
        for ref_i in range(len(reflections)):
            for ref_j in range(ref_i + 1, len(reflections)):
                contrast_output_str = self.contrasting_output_template
                contrast_output_str = contrast_output_str.replace("##REFLECTION_A##", str(ref_i + 1))
                contrast_output_str = contrast_output_str.replace("##REFLECTION_B##", str(ref_j + 1))
                # contrast_output_str = contrast_output_str.replace("##CONTRAST_ID##", str(len(contrasting_str) + 1))
                contrasting_str.append(contrast_output_str)

        prompt = self.contrasting_template
        prompt = prompt.replace("##REFLECTIONS##", "\n".join(reflections_str))
        prompt = prompt.replace("##CONTRASTS##", "\n".join(contrasting_str))
        _, response, _ = model.understand(prompt)

        mark = 'diff'
        differences, checklist = [], []
        for line in response.split("\n"):
            if self.args.language == 'zh':
                if line.startswith('评估') and mark == 'diff':
                    differences.append(line)
                elif line.startswith('检查清单') or mark == 'cklst':
                    mark = 'cklst'
                    checklist.extend(line.replace('检查清单', '').split('；'))
            else:
                if line.startswith('Evaluation') and mark == 'diff':
                    differences.append(line)
                elif line.startswith('Checklist') or mark == 'cklst':
                    mark = 'cklst'
                    checklist.extend(line.replace('Checklist', '').split(';'))

        return differences, checklist, reflections_str

    def summary(self, question, answer, reflections_str, differences, checklist, model):
        checklist = [re.sub(r'[;:；：]', '', c) for c in checklist]
        prompt = self.summary_template
        prompt = prompt.replace("##QUESTION##", question)
        prompt = prompt.replace("##ANSWER##", answer)
        prompt = prompt.replace("##REFLECTIONS##", "\n".join(reflections_str))
        prompt = prompt.replace("##DIFFERENCES##", "\n".join(differences))
        prompt = prompt.replace("##CHECKLIST##", "\n".join(checklist))
        _, summary, _ = model.understand(prompt)
        return summary

    def formalization(self, question, answer, reflections, summary, model):
        probs = []
        
        if self.args.language == 'zh':
            summaries = re.findall(r"评估\d+.*：\n?(.*)", summary)
        else:
            summaries = re.findall(r"Evaluation\d+.*：\n?(.*)", summary)
        summaries = summaries[:len(reflections)]
        assert len(summaries) == len(reflections), summaries

        for summary in summaries:
            prompt = self.summary_formalization_template
            prompt = prompt.replace("##QUESTION##", question)
            prompt = prompt.replace("##ANSWER##", answer)
            prompt = prompt.replace("##SUMMARY##", summary)

            def post_process(prob, response, output):
                if self.summary_formalization_targets[0] in response:
                    return 0, response, output
                elif self.summary_formalization_targets[1] in response:
                    return 1, response, output
                elif self.summary_formalization_targets[2] in response:
                    return 0.5, response, output
                else:
                    raise ValueError()

            prob, _, _ = model.understand(query=prompt, post_process=post_process)
            print(prob)
            probs.append(prob)

        return probs

    def __call__(self, sample, model, **kwargs):
        question = sample['question'].replace("\n", "")
        answer = sample['answer'].replace("\n", "")

        # reflection
        reflections, embeddings = self.refection(question, answer, model)

        # clustering
        indices = self.clustering(embeddings)
        reflections = [reflections[i].replace("\n", "") for i in indices]

        # contrasting
        differences, checklist, reflections_str = self.contrasting(reflections, model)

        # summary
        summary = self.summary(question, answer, reflections_str, differences, checklist, model)

        # summary formalization
        probs = self.formalization(question, answer, reflections, summary, model)

        probs = inverse_softmax(np.mean(probs))
        probs = np.tile(probs, (len(sample['sentences']), 1))
        return probs, ["" for _ in range(len(sample['sentences']))]


if __name__ == '__main__':
    parser.add_argument("--k", type=int, default=3)
    args = parse_args(parser)
    init_env(args, "SelfContrast")

    method = SelfContrast(args)
    evaluator = Evaluator(args)
    metrics = evaluator.evaluate(method)
    save_metrics(args.save_folder, metrics)
