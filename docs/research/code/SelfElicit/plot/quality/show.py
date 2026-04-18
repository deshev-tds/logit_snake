from collections import defaultdict
import numpy as np
import json
import matplotlib.pyplot as plt
import os

y_mean_all = {
    "MedHalluQA": [
        {'FaR': [0.8425, 0.8496, 0.8524, 0.8574, 0.8568, 0.8543, 0.8618],
         'IO': [0.9061, 0.8875, 0.8772, 0.9043, 0.883, 0.9239, 0.8849],
         'context': [0.9063, 0.8935, 0.9118, 0.9016, 0.8952, 0.9059, 0.8759],
         'elicit': [0.9063, 0.8969, 0.916, 0.9094, 0.9016, 0.9218, 0.9058]},
        {'FaR': [0.0183, 0.0212, 0.0175, 0.0174, 0.016, 0.0153, 0.0082],
         'IO': [0.0518, 0.0686, 0.0686, 0.0803, 0.0652, 0.0777, 0.0708],
         'context': [0.053, 0.0383, 0.0378, 0.0426, 0.0394, 0.0434, 0.0381],
         'elicit': [0.0524, 0.0472, 0.0445, 0.0455, 0.037, 0.0412, 0.0382]}
    ],
    "MedHalluQA_en": [
        {'FaR': [0.9194, 0.9212, 0.9251, 0.9266, 0.9239, 0.9208, 0.9269],
         'IO': [0.9416, 0.9374, 0.9358, 0.9367, 0.9391, 0.9363, 0.933],
         'context': [0.9416, 0.9379, 0.9362, 0.9331, 0.9453, 0.9407, 0.9396],
         'elicit': [0.9416, 0.9464, 0.9475, 0.9480, 0.9421, 0.9414, 0.9413]},
        {'FaR': [0.1092, 0.1149, 0.1115, 0.1024, 0.1069, 0.1077, 0.0894],
         'IO': [0.4449, 0.4133, 0.4127, 0.3944, 0.3689, 0.3824, 0.3921],
         'context': [0.4452, 0.4507, 0.4318, 0.3912, 0.376, 0.3602, 0.3618],
         'elicit': [0.4457, 0.3808, 0.3168, 0.2848, 0.2595, 0.2778, 0.292]}
    ],
    "WikiBio": [
        {'FaR': [0.6708, 0.677, 0.7359, 0.7299, 0.7463, 0.7697, 0.7417],
         'IO': [0.7571, 0.7368, 0.6442, 0.7007, 0.6596, 0.6917, 0.7308],
         'context': [0.7571, 0.7201, 0.6743, 0.7063, 0.6935, 0.7298, 0.6672],
         'elicit': [0.7571, 0.7862, 0.7358, 0.7856, 0.754, 0.7482, 0.7479]},
        {'FaR': [0.0191, 0.0064, 0.0131, 0.0103, 0.0095, 0.0078, 0.011],
         'IO': [0.2654, 0.3312, 0.2861, 0.2719, 0.2984, 0.2667, 0.3029],
         'context': [0.2654, 0.2652, 0.2126, 0.2233, 0.1936, 0.2337, 0.2074],
         'elicit': [0.2654, 0.231, 0.2242, 0.2044, 0.1914, 0.1947, 0.2046]}
    ]
}
y_std_all = {
    "MedHalluQA": [
        {'FaR': [0.1674, 0.1625, 0.1614, 0.1639, 0.147, 0.1719, 0.1714],
         'IO': [0.1931, 0.225, 0.2288, 0.2002, 0.2535, 0.1773, 0.237],
         'context': [0.1805, 0.2082, 0.1694, 0.1808, 0.1838, 0.1677, 0.1946],
         'elicit': [0.2019, 0.2113, 0.1742, 0.1943, 0.1982, 0.171, 0.2191]},
        {'FaR': [0.045, 0.0491, 0.0432, 0.0426, 0.041, 0.0434, 0.0276],
         'IO': [0.1561, 0.1869, 0.1601, 0.1794, 0.1627, 0.1836, 0.134],
         'context': [0.1208, 0.1081, 0.1138, 0.1215, 0.1104, 0.1202, 0.11],
         'elicit': [0.1215, 0.1227, 0.1196, 0.1154, 0.1024, 0.1138, 0.1211]}

    ],
    "MedHalluQA_en": [
        {'FaR': [0.1021, 0.1044, 0.0963, 0.0990, 0.1236, 0.0922, 0.1021],
         'IO': [0.1539, 0.1213, 0.1334, 0.1025, 0.1242, 0.1215, 0.0968],
         'context': [0.1361, 0.1657, 0.1674, 0.1596, 0.1470, 0.1663, 0.1784],
         'elicit': [0.1246, 0.1399, 0.1316, 0.1390, 0.1317, 0.1437, 0.1334]},
        {'FaR': [0.1233, 0.1204, 0.1251, 0.1140, 0.1131, 0.1129, 0.0925],
         'IO': [0.2787, 0.2523, 0.2582, 0.2651, 0.2586, 0.2571, 0.2508],
         'context': [0.3467, 0.3224, 0.3295, 0.3267, 0.2988, 0.3125, 0.3063],
         'elicit': [0.3025, 0.2886, 0.2943, 0.2811, 0.2937, 0.2653, 0.2751]}
    ],
    "WikiBio": [
        {'FaR': [0.2033, 0.2121, 0.1989, 0.206, 0.2152, 0.1888, 0.2063],
         'IO': [0.3348, 0.3834, 0.3937, 0.3905, 0.3826, 0.3841, 0.3786],
         'context': [0.3332, 0.3244, 0.3581, 0.325, 0.337, 0.3429, 0.333],
         'elicit': [0.4031, 0.3426, 0.3625, 0.3325, 0.3517, 0.3632, 0.3281]},
        {'FaR': [0.0418, 0.0258, 0.0357, 0.0328, 0.0308, 0.0294, 0.0368],
         'IO': [0.2943, 0.3703, 0.3439, 0.3611, 0.3747, 0.3524, 0.3798],
         'context': [0.3, 0.2827, 0.2717, 0.2923, 0.2894, 0.2861, 0.2592],
         'elicit': [0.3081, 0.2937, 0.2739, 0.2538, 0.2761, 0.2926, 0.3105]}
    ]
}

MAX = 7
fontsize = 15
path = {
    'fact': {'FaR': 'fact-FaR.txt',
             'IO': 'fact-io.txt',
             'context': 'fact-context.txt',
             'elicit': 'fact-elicit.txt'},
    'rel': {'FaR': 'rel-FaR.txt',
            'IO': 'rel-io.txt',
            'context': 'rel-context.txt',
            'elicit': 'rel-elicit.txt'}
}
y_axis = [r'Factuality $\uparrow$', r'Diversity $\uparrow$']
colors = plt.colormaps['Set3'].colors
models = ['FaR', 'IO', 'context', 'elicit']
model_names = {"FaR": 'CoT', "IO": 'Reflect', "context": 'Context', "elicit": 'Elicit'}


def load_data(dataset):
    dir = os.path.join("./output/", dataset)
    fact_results = defaultdict(list)
    for k in path['fact']:
        with open(os.path.join(dir, path['fact'][k]), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                fact_results[k].append(json.loads(line))

    rel_results = defaultdict(list)
    for k in path['rel']:
        with open(os.path.join(dir, path['rel'][k]), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                rel_results[k].append(json.loads(line))

    return fact_results, rel_results


def get_mean_std(dataset):
    if len(y_mean_all[dataset]) > 0 and len(y_mean_all[dataset][i]) > 0:
        y_mean = y_mean_all[dataset][i]
        y_std = y_std_all[dataset][i]

    else:
        results = load_data(dataset)

        y_mean, y_std = defaultdict(list), defaultdict(list)
        for k in results[i]:
            for j in range(MAX):
                temp = [ele[j] for ele in results[i][k] if len(ele) > j]
                temp = [np.mean(ele) for ele in temp if ele]
                y_mean[k].append(np.mean(temp))
                y_std[k].append(np.std(temp))

        print(f"y_mean for figure {i}", y_mean)
        print(f"y_std for figure {i}", y_std)

    return y_mean, y_std


if __name__ == '__main__':

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    for fid, dataset in enumerate(['MedHalluQA', 'MedHalluQA_en', 'WikiBio']):
        supax = axs[fid]
        supax.axis('off')
        supax.set_title([r"$\bf(a)$ MedHallu-zh", r"$\bf(b)$ MedHallu-en", r"$\bf(c)$ WikiBio"][fid],
                        y=-0.25, fontsize=fontsize + 6)
        gs = supax.get_gridspec()
        subgs = gs[fid].subgridspec(1, 2, wspace=0.14)

        for i in range(2):
            ax = fig.add_subplot(subgs[i])
            y_mean, y_std = get_mean_std(dataset)

            x = range(MAX)
            ax.set_xticks(x, np.arange(1, 1 + MAX))
            ax.tick_params(labelsize=fontsize)

            for model in models:
                avg = np.array(y_mean[model])
                avg = 1 - avg if i == 1 else avg
                std = np.array(y_std[model]) / 50
                lower = avg - 2 * std
                upper = avg + 2 * std
                print("|", ["fact", "rel"][i], "|", model, "|", " | ".join(f"{_:.3f}" for _ in avg), "|")
                ax.plot(x, avg, 'o-', label=model_names[model] if i == 0 and fid == 0 else None, linewidth=3, markersize=10)
                ax.fill_between(x, lower, upper, alpha=0.2)

    fig.legend(loc="upper center", fontsize=fontsize + 6, ncols=4, columnspacing=3)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.068, top=0.93, right=0.995, left=0.06, wspace=0.13, hspace=0.3)
    plt.savefig(f"./ExpElicit.pdf", dpi=300)
    plt.show()
