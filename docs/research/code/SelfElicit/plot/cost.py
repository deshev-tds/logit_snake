import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.lines import Line2D

data = {
    "SelfElicit": [[0.8098, 0.6712, 12452, 1020170],
                   [0.8032, 0.6557, 11845, 1404662],
                   [0.5936, 0.6534, 5002, 538163],
                   [0.7889, 0.564, 2280, 274958]],
    "IO": [[0.7705, 0.598, 7274, 390429],
           [0.7619, 0.6218, 7422, 636621],
           [0.5271, 0.6283, 1908, 159091],
           [0.729, 0.5162, 1863, 113726]],
    "ContextIO": [[0.7604, 0.6031, 7552, 398882],
                  [0.733, 0.5822, 7552, 119244],
                  [0.5869, 0.5221, 1908, 175185],
                  [0.68, 0.4595, 1863, 148577]],
    "HistoryIO": [[0.7816, 0.6531, 7552, 370379],
                  [0.7807, 0.6143, 7552, 190890],
                  [0.5431, 0.6139, 1908, 219246],
                  [0.78, 0.5315, 1863, 153968]],
    "CoT": [[0.6378, 0.5708, 7274, 933532],
            [0.5958, 0.57, 7422, 1295719],
            [0.5002, 0.5664, 1908, 526725],
            [0.5986, 0.5156, 1863, 559563]],
    "FaR": [[0.7625, 0.6134, 14104, 2309398],
            [0.7429, 0.6298, 14104, 2751656],
            [0.5425, 0.5081, 5724, 1399236],
            [0.7565, 0.5566, 5589, 2244516]],
    "SelfCheckGPT": [[0.5, 0.5, 130912, 13711425],
                     [0.6818, 0.6232, 131066, 10828495],
                     [0.539, 0.6389, 10730, 870249],
                     [0.6307, 0.5241, 11400, 1662209]],
    "ChatProtect": [[0.5118, 0.517, 138758, 5703176],
                    [0.505, 0.5052, 164010, 6398109],
                    [0.5124, 0.6574, 86178, 2410890],
                    [0.5278, 0.4943, 60240, 1047074]],
    "CoVE": [[0.5971, 0.5475, 36852, 1828383],
             [0.4998, 0.4984, 38696, 2483669],
             [0.5266, 0.5242, 12619, 1156417],
             [0.6722, 0.5133, 9641, 1823669]],
}
data = OrderedDict(data)
titles = [r'$\bf(a)$MedHallu-zh', r'$\bf(b)$MedHallu-en', r'$\bf(c)$WikiBio', r'$\bf(d)$HaluEval2']

size = 500
base = 50
fontsize = 15

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))
    cmap = plt.colormaps['tab10']

    for ds in range(4):
        ax = axs[ds]

        all_performance = [(e[ds][0] + e[ds][2]) / 1 for e in data.values()]
        all_calls = [e[ds][2] for e in data.values()]
        all_tokens = [e[ds][3] for e in data.values()]
        max_performance, min_performance = max(all_performance), min(all_performance)
        max_calls, min_calls = max(all_calls), min(all_calls)
        max_tokens, min_tokens = max(all_tokens), min(all_tokens)

        for model, values in data.items():
            values = values[ds]
            performance = (values[0] + values[1]) / 2
            calls = (values[2] / min_calls) * size + base
            tokens = values[3]
            color = cmap(list(data.keys()).index(model) / len(data))

            scatter = ax.scatter(performance, tokens, s=calls,
                                 color=color,
                                 alpha=0.7,
                                 label=model if ds == 0 else None)
            if model == 'SelfElicit':
                pass
                # ax.axvline(x=performance, linestyle="--", color='gray')
                # ax.axhline(y=tokens,  linestyle="--", color='gray')

            ax.set_title(titles[ds], fontsize=fontsize + 4, y=-0.36)
            ax.set_xlabel('avg. AUC' + r"$\uparrow$", fontsize=fontsize)
            if ds == 0:
                pass
                # ax.set_ylabel('#tokens (log)' + r"$\downarrow$", fontsize=fontsize, labelpad=0)
            ax.set_yscale("log")
            ax.tick_params(labelsize=fontsize)

    legend_marker_size = 200
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=k,
                              markerfacecolor=cmap(list(data.keys()).index(k) / len(data)),
                              markersize=np.sqrt(legend_marker_size))
                       for k in data.keys()]

    fig.legend(handles=legend_elements, loc="upper center", fontsize=fontsize + 4, ncols=5, columnspacing=1)

    fig.tight_layout()
    plt.subplots_adjust(top=0.78, bottom=0.205, left=0.038, right=0.995, wspace=0.2, hspace=0.37)
    plt.savefig(f"./ExpCost.pdf", dpi=300)
    plt.show()
