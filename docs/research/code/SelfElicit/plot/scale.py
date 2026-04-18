import re
import matplotlib.pyplot as plt

data = {
    "IO": {
        0.5: [2, "12分19秒", 0.15, 0.2899, 0.7255, 0.4062, 0.627, 0.5517],
        1.8: [2, "3小时1分33秒", 0.1668, 0.2893, 0.7423, 0.4138, 0.6283, 0.5641],
        4: [2, "2小时10分3秒", 0.1835, 0.32, 0.7724, 0.4362, 0.6424, 0.605],
        7: [2, "2小时5分30秒", 0.1866, 0.3187, 0.7705, 0.4409, 0.6369, 0.598],
        14: [4, "2小时25分45秒", 0.2349, 0.3521, 0.8022, 0.4625, 0.6653, 0.6627],
        32: [4, "7小时53分9秒", 0.2644, 0.3636, 0.8137, 0.492, 0.6686, 0.6919],
        110: [8, "1天1小时56分36秒", 0.2837, 0.387, 0.817, 0.4578, 0.6584, 0.6998]
    },
    # "ConfScore": {
    #     0.5: [2, "1小时2分16秒", 0.1281, 0.2599, 0.6432, 0.3996, 0.6203, 0.5139],
    #     1.8: [2, "3小时19分43秒", 0.1319, 0.2726, 0.6888, 0.4012, 0.6262, 0.5253],
    #     4: [2, "1小时24分57秒", 0.1531, 0.2687, 0.6782, 0.4118, 0.625, 0.5697],
    #     7: [2, "2小时4分45秒", 0.1825, 0.275, 0.726, 0.4012, 0.6262, 0.5793],
    #     14: [4, "2小时36分12秒", 0.2219, 0.3094, 0.7549, 0.4538, 0.6274, 0.6431],
    #     32: [4, "8小时36分7秒", 0.232, 0.3353, 0.771, 0.4482, 0.6289, 0.6594],
    #     110: [8, "1天4小时5分16秒", 0.231, 0.3407, 0.7846, 0.4594, 0.6424, 0.6544]
    # },
    "CoT": {
        0.5: [2, "1小时46分12秒", 0.1428, 0.2661, 0.6505, 0.4095, 0.6205, 0.5618],
        1.8: [2, "3小时38分34秒", 0.132, 0.2687, 0.6742, 0.4041, 0.6203, 0.5429],
        4: [2, "2小时53分47秒", 0.1307, 0.2431, 0.6233, 0.3983, 0.6203, 0.5335],
        7: [2, "4小时13分24秒", 0.1916, 0.2815, 0.6378, 0.4024, 0.6203, 0.5708],
        14: [4, "5小时6分49秒", 0.1903, 0.3089, 0.6744, 0.4396, 0.6205, 0.6079],
        32: [4, "14小时53分5秒", 0.1328, 0.2737, 0.6974, 0.4083, 0.6203, 0.6055],
        110: [8, "68小时6分43秒", 0.1361, 0.2789, 0.68, 0.4012, 0.625, 0.6046]
    },
    "FaR": {
        0.5: [2, "3小时39分9秒", 0.1385, 0.2797, 0.7019, 0.402, 0.627, 0.5238],
        1.8: [2, "21小时23分58秒", 0.163, 0.2921, 0.7372, 0.4286, 0.6277, 0.581],
        4: [2, "5小时55分32秒", 0.1528, 0.2863, 0.7291, 0.4095, 0.6294, 0.5525],
        7: [4, "14小时34分49秒", 0.2065, 0.2955, 0.7625, 0.4408, 0.635, 0.6134],
        14: [4, "19小时31分56秒", 0.2174, 0.3272, 0.7847, 0.4664, 0.6577, 0.6533],
        32: [4, "2天17小时9分11秒", 0.2042, 0.3289, 0.7834, 0.4450, 0.6501, 0.6448],
        110: [8, "175小时5分16秒", 0.2637, 0.3550, 0.8065, 0.4753, 0.6488, 0.6837]
    },
    "SelfElicit": {
        0.5: [2, "55分43秒", 0.163, 0.3, 0.7366, 0.4059, 0.6266, 0.5279],
        1.8: [2, "5小时18分9秒", 0.186, 0.3253, 0.7678, 0.4258, 0.6307, 0.6026],
        4: [2, "39分23秒", 0.2323, 0.3413, 0.7849, 0.4369, 0.6246, 0.625],
        7: [2, "4小时26分41秒", 0.269, 0.3637, 0.8098, 0.4745, 0.6433, 0.6712],
        14: [4, "5小时19分56秒", 0.2678, 0.3863, 0.8121, 0.4639, 0.6403, 0.6705],
        32: [4, "8小时45分20秒", 0.2938, 0.409, 0.8288, 0.4911, 0.6643, 0.7134],
        110: [8, "64小时40分53秒", 0.2872, 0.4064, 0.8298, 0.4828, 0.6625, 0.713]
    }
}


def get_time_spent(n_core, time):
    pattern = r"((?P<d>\d+)天)?((?P<h>\d+)小时)?((?P<m>\d+)分)?((?P<s>\d+)秒)?"

    # 使用正则表达式解析输入的字符串
    match = re.match(pattern, time)
    days = int(match.group('d') or 0)
    hours = int(match.group('h') or 0)
    minutes = int(match.group('m') or 0)
    seconds = int(match.group('s') or 0)
    total_seconds = days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds

    total_seconds = n_core * total_seconds
    total_hours = total_seconds / 60 / 60
    return round(total_hours, 2)


fontsize = 15
linewidth = 2.0

if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for model in data.keys():
        sizes = [0.5, 1.8, 4, 7, 14, 32, 110]
        p = [(data[model][s][4] + data[model][s][7]) / 2 for s in sizes]
        cost = [get_time_spent(data[model][s][0], data[model][s][1]) for s in sizes]

        axs[0].plot(sizes, p, label=model, alpha=0.9, linewidth=linewidth)
        axs[0].set_xscale('log')
        axs[0].set_xticks(sizes)
        axs[0].set_xticklabels(sizes)
        axs[0].set_title(r"$\bf(a)$" + "Model Size vs Performance", fontsize=fontsize + 4, y=-0.4)
        axs[0].tick_params(labelsize=fontsize)
        axs[0].set_xlabel('Model Size (B)', fontsize=fontsize)
        axs[0].set_ylabel('avg. AUC', fontsize=fontsize)

        axs[1].plot(sizes, cost, alpha=0.9, linewidth=linewidth)
        axs[1].set_xscale('log')
        axs[1].set_xticks(sizes)
        axs[1].set_xticklabels(sizes)
        axs[1].set_yscale('log')
        axs[1].set_title(r"$\bf(b)$" + "Model Size vs Cost", fontsize=fontsize, y=-0.4)
        axs[1].tick_params(labelsize=fontsize)
        axs[1].set_xlabel('Model Size (B)', fontsize=fontsize)
        axs[1].set_ylabel('Cost (core hours)', fontsize=fontsize)

        for s in sizes:
            axs[0].axvline(x=s, linestyle='--', color='grey', linewidth=0.5, alpha=0.5)
            axs[1].axvline(x=s, linestyle='--', color='grey', linewidth=0.5, alpha=0.5)

    fig.legend(loc="upper center", fontsize=fontsize + 4, ncols=4, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.83, wspace=0.2, bottom=0.25)
    plt.savefig("./ExpScale.pdf", dpi=300)
    plt.show()
