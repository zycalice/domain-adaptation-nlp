import json
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    with open('../outputs/accuracies_ti_all_domains_conf1_lr_c0.1_l2_fs1.json') as f:
        accuracies_ti = json.load(f)

    print(accuracies_ti)

    t1_accuracies = {}
    for exp in accuracies_ti:
        t1_accuracies_exp = []
        for group in accuracies_ti[exp]:
            if group != "no_self_train_adaptation_score":
                t1_accuracies_exp.append(accuracies_ti[exp][group][0])
            else:
                t1_accuracies_exp.append(accuracies_ti[exp][group])
        t1_accuracies[exp] = t1_accuracies_exp

    print(t1_accuracies)

    tn_accuracies = {}
    for exp in accuracies_ti:
        tn_accuracies_exp = []
        for group in accuracies_ti[exp]:
            if group != "no_self_train_adaptation_score":
                tn_accuracies_exp.append(accuracies_ti[exp][group][-1])
            else:
                tn_accuracies_exp.append(accuracies_ti[exp][group])
        tn_accuracies[exp] = tn_accuracies_exp

    print(tn_accuracies)

    exps = [
        'tw_to_az', 'tw_to_mv',
        'az_to_tw', 'az_to_mv',
        'mv_to_tw', 'mv_to_az',
        ]

    # t1
    for exp in exps:
        to_graph = np.array(t1_accuracies[exp]) - t1_accuracies[exp][0]
        plt.plot(to_graph)
    plt.plot([0 for i in range(11)], linestyle='dashed', color="black")
    plt.ylim([-0.05, 0.05])
    plt.legend(exps)
    plt.title("self train on t1 accuracies by fraction of t")
    plt.show()

    # tn (final)
    for exp in exps:
        to_graph = np.array(tn_accuracies[exp]) - tn_accuracies[exp][0]
        plt.plot(to_graph)
    plt.plot([0 for i in range(11)], linestyle='dashed', color="black")
    plt.ylim([-0.05, 0.05])
    plt.legend(exps)
    plt.title("gradual self train accuracies by number of clusters/groups")
    plt.show()
