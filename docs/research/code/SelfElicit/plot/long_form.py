import os
import pickle
from collections import defaultdict

from data.dataloader import load_dataset
from utils.metrics import get_metrics
import numpy as np
import matplotlib.pyplot as plt

# Model qwen1.5-7b-chat
models = ["SelfElicit", "IO", "ContextIO", "HistoryIO", "CoT", "FaR", "SelfCheckGPT", "ChatProtect", "CoVE"]
markers = {"SelfElicit": "*", "IO": "^", "ContextIO": "o", "HistoryIO": "o", "CoVE": "^",
           "SelfCheckGPT": "^", "ChatProtect": "^", "CoT": "^", "FaR": "^"}
saves = {
    "MedHallu": {
        "SelfElicit": "../saves/thought/MedHallu/qwen1_5_7b_chat-sentence-sent-selfkg-0823",
        "IO": "../saves/IO/MedHallu/qwen1_5_7b_chat-sentence",
        "ContextIO": "../saves/IO/MedHallu/qwen1_5_7b_chat-sentence-context",
        "HistoryIO": "../saves/IO/MedHallu/qwen1_5_7b_chat-sentence-history",
        "CoVE": "../saves/CoVE/MedHallu/qwen1_5_7b_chat-sentence",
        "FaR": "../saves/FaR/MedHallu/qwen1_5_14b_chat-sentence",
        "CoT": "../saves/CoT/MedHallu/qwen1_5_7b_chat-sentence-1",
        "SelfCheckGPT": "../saves/SelfCheckGPT/MedHallu/qwen1_5_7b_chat-sentence-prompt",
        "ChatProtect": "../saves/ChatProtect/MedHallu/qwen1_5_7b_chat-sentence-max",
    },
    "MedHallu_en": {
        "SelfElicit": "../saves/thought/MedHallu_en/qwen1_5_7b_chat-sentence-0830",
        "IO": "../saves/IO/MedHallu_en/qwen1_5_7b_chat-sentence",
        "ContextIO": "../saves/IO/MedHallu_en/qwen1_5_7b_chat-sentence-context",
        "HistoryIO": "../saves/IO/MedHallu_en/qwen1_5_7b_chat-sentence-history",
        "CoVE": "../saves/CoVE/MedHallu_en/qwen1_5_7b_chat-sentence",
        "FaR": "../saves/FaR/MedHallu_en/qwen2_5_7b_chat-sentence",
        "CoT": "../saves/CoT/MedHallu_en/qwen1_5_7b_chat-sentence-1",
        "SelfCheckGPT": "../saves/SelfCheckGPT/MedHallu_en/qwen1_5_7b_chat-sentence-prompt",
        "ChatProtect": "../saves/ChatProtect/MedHallu_en/qwen1_5_7b_chat-sentence-max",
    }
}


def get_result(dataset):
    performance_horizons = defaultdict(list)
    dataloader = load_dataset(f"../data/dataset/{dataset}", False)['test']

    for horizon in range(0, 11):
        print("*" * 30, horizon, "*" * 30)
        idx = [i for i, sample in enumerate(dataloader) if len(sample['sentences']) >= horizon]

        for model in saves[dataset]:
            with open(os.path.join(saves[dataset][model], 'output.pkl'), "rb") as f:
                cache = pickle.load(f)
                probs = cache["probs"]

            # print metrics
            labels = [sample['sentence_labels'] for sample in dataloader]
            sample_labels = [sample['label'] for sample in dataloader]
            metrics = get_metrics(probs=[probs[i] for i in idx],
                                  labels=[labels[i] for i in idx],
                                  sample_labels=[sample_labels[i] for i in idx],
                                  strategies=['ignore'],
                                  aggregate='max',
                                  verbose=False)

            performance_horizons[model].append(round(np.mean([metrics['ignore-auc'], metrics['ignore-auc-agg']]), 4))

    print(performance_horizons)


performance_horizons_all = {
    "MedHallu-zh":
        {'SelfElicit': [0.7405, 0.7405, 0.741, 0.7398, 0.744, 0.7411, 0.7333, 0.7442, 0.7537, 0.7592, 0.7613],
         'IO': [0.6926, 0.6926, 0.6912, 0.6903, 0.698, 0.6854, 0.6829, 0.6831, 0.6854, 0.6922, 0.6899],
         'ContextIO': [0.6917, 0.6917, 0.6904, 0.69, 0.6972, 0.6905, 0.686, 0.6893, 0.6905, 0.70, 0.7081],
         'HistoryIO': [0.7173, 0.7173, 0.7163, 0.7152, 0.7284, 0.7192, 0.7197, 0.73, 0.7299, 0.7372, 0.7393],
         'CoT': [0.5836, 0.5836, 0.5847, 0.5825, 0.584, 0.5776, 0.5794, 0.5832, 0.5736, 0.5756, 0.5834],
         'FaR': [0.688, 0.688, 0.6859, 0.6836, 0.6847, 0.6720, 0.6730, 0.6756, 0.6734, 0.6780, 0.6841],
         'CoVE': [0.5723, 0.5723, 0.5727, 0.57, 0.5641, 0.5615, 0.5625, 0.5658, 0.5698, 0.5663, 0.5674],
         'SelfCheckGPT': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
         'ChatProtect': [0.5144, 0.5144, 0.5144, 0.5127, 0.5141, 0.5072, 0.5102, 0.5144, 0.5202, 0.5204, 0.5203]},
    "MedHallu-en":
        {'SelfElicit': [0.7294, 0.7294, 0.7286, 0.7264, 0.7315, 0.7299, 0.7258, 0.7322, 0.7385, 0.742, 0.752],
         'IO': [0.6919, 0.6919, 0.6898, 0.6873, 0.6905, 0.6771, 0.6695, 0.6821, 0.6869, 0.6878, 0.6971],
         'ContextIO': [0.6883, 0.6883, 0.6862, 0.683, 0.688, 0.6806, 0.6854, 0.6931, 0.6945, 0.6934, 0.7139],
         'HistoryIO': [0.7191, 0.7191, 0.7173, 0.7136, 0.7209, 0.7088, 0.709, 0.7186, 0.7257, 0.7285, 0.7345],
         'CoT': [0.5715, 0.5715, 0.5719, 0.5705, 0.5821, 0.5817, 0.5826, 0.5934, 0.5985, 0.5894, 0.5943],
         'CoVE': [0.4991, 0.4991, 0.4991, 0.4991, 0.499, 0.4987, 0.4984, 0.498, 0.4988, 0.5, 0.5],
         'FaR': [0.6963, 0.6963, 0.6954, 0.6920, 0.6945, 0.6826, 0.6698, 0.6893, 0.6829, 0.6736, 0.683],
         'SelfCheckGPT': [0.6525, 0.6525, 0.6505, 0.6487, 0.6387, 0.6286, 0.6076, 0.6016, 0.5943, 0.5664, 0.5542],
         'ChatProtect': [0.5051, 0.5051, 0.5051, 0.5028, 0.5007, 0.498, 0.4993, 0.5001, 0.495, 0.4948, 0.4953]}
}
fontsize = 15

if __name__ == '__main__':
    # get_result("MedHallu")
    # get_result("MedHallu_en")
    # exit(0)

    # save output results
    # plot output results
    RANGE = range(5, 11)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, ds in enumerate(['MedHallu-zh', "MedHallu-en"]):
        axs[i].set_title([r"$\bf(a)$", r"$\bf(b)$"][i] + r"Rel. Perform on " + ds, y=-0.2, fontsize=fontsize + 6)
        axs[i].tick_params(labelsize=fontsize)
        axs[i].set_yticks([0.7, 0.8, 0.9, 1.0], ["70", "80", "90", "100"])
        for model in models:
            # y: avg AUC performance
            y = []
            for horizon in RANGE:
                standard = performance_horizons_all[ds]['SelfElicit'][horizon]
                score = performance_horizons_all[ds][model][horizon]
                value = score / standard
                y.append(value)

            axs[i].plot(RANGE, y, markers[model] + "-", label=model if i == 0 else None, linewidth=4, markersize=13)

    fig.legend(loc="upper center", fontsize=fontsize + 6, ncols=5, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.78, bottom=0.13, wspace=0.1, hspace=0.35, left=0.045)
    plt.savefig("./ExpLongform.pdf", dpi=300)
    plt.show()
