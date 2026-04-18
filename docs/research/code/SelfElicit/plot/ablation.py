import numpy as np
import matplotlib.pyplot as plt

results = {
    "MedHallu-zh": {
        "Qwen": {
            "w/o context": [0.2514, 0.7433, 0.4438, 0.6211],
            "w/o elicit": [0.252, 0.7961, 0.4409, 0.6154],
            "w/o sample": [0.2618, 0.8028, 0.4519, 0.6523],
            "w/o conflict": [0.2655, 0.808, 0.4697, 0.6679],
            "full": [0.269, 0.8098, 0.4745, 0.6712]
        },
        "ChatGLM": {
            "w/o context": [0.2119, 0.7871, 0.4418, 0.6131],
            "w/o elicit": [0.2149, 0.7891, 0.442, 0.6138],
            "w/o sample": [0.215, 0.788, 0.438, 0.615],
            "w/o conflict": [0.219, 0.7878, 0.4408, 0.615],
            "full": [0.2282, 0.7984, 0.4454, 0.6221]
        }
    },
    "MedHallu-en": {
        "Qwen": {
            "w/o context": [0.2312, 0.7845, 0.4405, 0.6215],
            "w/o elicit": [0.2349, 0.7955, 0.4422, 0.624],
            "w/o sample": [0.236, 0.797, 0.4523, 0.632],
            "w/o conflict": [0.236, 0.7956, 0.4515, 0.629],
            "full": [0.242, 0.8032, 0.4628, 0.6557]
        },
        "Llama2": {
            "w/o context": [0.145, 0.7123, 0.4099, 0.5716],
            "w/o elicit": [0.1701, 0.7278, 0.4083, 0.5693],
            "w/o sample": [0.1704, 0.7269, 0.4049, 0.5636],
            "w/o conflict": [0.1692, 0.7188, 0.4063, 0.5717],
            "full": [0.1809, 0.7477, 0.4083, 0.5815]
        }},
    "WikiBio": {
        "Qwen": {
            "w/o context": [0.527, 0.628],
            "w/o elicit": [0.547, 0.622],
            "w/o sample": [0.543, 0.614],
            "w/o conflict": [0.562, 0.632],
            "full": [0.594, 0.653]
        },
        "Llama2": {
            "w/o context": [0.516, 0.559],
            "w/o elicit": [0.521, 0.572],
            "w/o sample": [0.527, 0.541],
            "w/o conflict": [0.542, 0.639],
            "full": [0.568, 0.705]
        }},
    "HaluEval2": {
        "Qwen": {
            "w/o context": [0.729, 0.516],
            "w/o elicit": [0.723, 0.510],
            "w/o sample": [0.780, 0.532],
            "w/o conflict": [0.781, 0.542],
            "full": [0.789, 0.564]
        },
        "Llama2": {
            "w/o context": [0.513, 0.529],
            "w/o elicit": [0.512, 0.523],
            "w/o sample": [0.516, 0.526],
            "w/o conflict": [0.518, 0.531],
            "full": [0.521, 0.545]
        }}
}
metrics = ['AUC']
width = .15
colors = plt.colormaps['Set3'].colors
hatches = ['//', 'x', 'x', 'x', '\\\\', '\\\\', '++']  # '++', '*', 'O', 'o', '.', '/'
fontsize = 20


def draw(axs, pos, all_data, title):
    ax = axs[pos[0], pos[1]]
    ax.set_title(title, fontsize=fontsize + 12, y=-0.2)
    ax.axis('off')

    gs = ax.get_gridspec()
    subgs = gs[pos[0], pos[1]].subgridspec(1, 2, wspace=0.45)

    for i, sax in enumerate(subgs):
        model = list(all_data.keys())[i]
        data = all_data[model]

        # plot sentence AUC
        subax = fig.add_subplot(sax)
        subax.set_title(model, fontsize=fontsize + 6)
        subax.set_xticks(range(2), ["S", "R"])
        subax.tick_params(labelsize=fontsize)
        if pos[1] == 0 and i == 0:
            pass
            # subax.set_ylabel('AUC', fontsize=fontsize + 6, labelpad=-10.0)

        min_val, max_val = 100, 0
        for k, variate in enumerate(data):
            x = (k - len(data) / 2) * width
            if len(data[variate]) == 2:
                y = data[variate][0]
            elif len(data[variate]) == 4:
                y = data[variate][1]
            else:
                raise ValueError

            min_val = min(min_val, y)
            max_val = max(max_val, y)
            subax.bar(x, y, width * 0.9, color=colors[k + 2], hatch=hatches[k])

        subax.set_ylim(min_val - (max_val - min_val) / 2, max_val + (max_val - min_val) / 10)
        yticks = list(set([round(_, 2) for _ in subax.get_yticks()][1:-1]))  # 2 digits
        subax.set_yticks(yticks)

        # plot response AUC
        subax = subax.twinx()
        subax.tick_params(labelsize=fontsize)
        if pos[1] == 1 and i == 1:
            pass
            # subax.set_ylabel('AUC', fontsize=fontsize + 6)

        min_val, max_val = 100, 0
        for k, variate in enumerate(data):
            x = 1 + (k - len(data) / 2) * width
            if len(data[variate]) == 2:
                y = data[variate][1]
            elif len(data[variate]) == 4:
                y = data[variate][3]
            else:
                raise ValueError

            min_val = min(min_val, y)
            max_val = max(max_val, y)
            label = variate if (pos[0] == 0 and pos[1] == 0 and i == 0) else None
            subax.bar(x, y, width * 0.9, color=colors[k + 2], hatch=hatches[k], label=label)

        subax.set_ylim(min_val - (max_val - min_val) * 2, max_val + (max_val - min_val) / 10)
        yticks = list(set([round(_, 2) for _ in subax.get_yticks()][1:-1]))  # 2 digits
        subax.set_yticks(yticks)


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    draw(axs, [0, 0], results['MedHallu-zh'], r"$\bf(a)$" + " MedHallu-zh")
    draw(axs, [0, 1], results['MedHallu-en'], r"$\bf(b)$" + " MedHallu-en")
    draw(axs, [1, 0], results['WikiBio'], r"$\bf(c)$" + " WikiBio")
    draw(axs, [1, 1], results['HaluEval2'], r"$\bf(d)$" + " HaluEval2")

    fig.legend(loc="upper center", fontsize=fontsize + 12, ncols=5, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.07, wspace=0.17, hspace=0.3, left=0.04, right=0.96)
    plt.savefig("./ExpAblation.pdf", dpi=300)
    plt.show()

    # output relative results

    for variant in ["w/o context", "w/o elicit", "w/o sample", "w/o conflict", "full"]:
        if variant == 'full':
            continue

        print(variant)

        out = []
        for dataset in results:
            for model in results[dataset]:
                full_res = results[dataset][model]['full']
                variant_res = results[dataset][model][variant]

                if len(full_res) == 2:
                    out.append(full_res[0] / variant_res[0])
                    out.append(full_res[1] / variant_res[1])
                elif len(full_res) == 4:
                    out.append(full_res[1] / variant_res[1])
                    out.append(full_res[3] / variant_res[3])
                else:
                    raise ValueError

        print(np.mean(out) - 1)
