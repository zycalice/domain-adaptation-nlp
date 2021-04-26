import pickle
import json
from utils import *
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    with open("../data/amazon_reviews/amazon_4.pickle", "rb") as fr:
        all_data = pickle.load(fr)

    # initiate.
    lr = LogisticRegression(C=0.1, max_iter=200000)
    subsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    self_train_groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    accuracies_ti_all_domains = {}
    final_accuracies_all_domains = {}
    dists_all_domains = {}
    data_size = 2000

    for source in all_data:
        for target in all_data:
            train_name = source[2] + "_to_" + target[2]
            print("\n", train_name)

            final_accuracies, accuracies_ti, dists = run_gradual_train_ranges(
                x_source_raw=source[0], y_source_raw=source[1],
                x_target_raw=target[0], y_target_raw=target[1],
                base_model=lr, data_size=data_size, group_range=self_train_groups, plot_hist=False,
                dist_type="cos", conf=0.1, subset_range=None, few_shot_size=0.0025, label_final=False,
            )

            final_accuracies_all_domains[train_name] = final_accuracies
            accuracies_ti_all_domains[train_name] = accuracies_ti
            dists_all_domains[train_name] = dists

    print(accuracies_ti_all_domains)
    with open("../outputs/accuracies_ti_amazon_conf1_c0.1_fs0.0025.json", "w") as outfile:
        json.dump(accuracies_ti_all_domains, outfile, indent=4)

    # # final label with conf
    # scores = {}
    # for source in all_data:
    #     for target in all_data:
    #         train_name = source[2] + "_to_" + target[2]
    #         print("\n", train_name)
    #
    #         scores[train_name] = run_gradual_train_final_label_conf_groups(
    #             x_source_raw=source[0], y_source_raw=source[1],
    #             x_target_raw=target[0], y_target_raw=target[1],
    #             base_model=lr, data_size=data_size, conf=0.3,
    #             few_shot_size=0
    #         )
    #
    #         # final_accuracies_all_domains[train_name] = final_accuracies
    #         # accuracies_ti_all_domains[train_name] = accuracies_ti
    #         # dists_all_domains[train_name] = dists
    #
    # print(scores)
    # with open("../outputs/accuracies_ti_amazon_finaL_label_conf0.3_c0.1.json", "w") as outfile:
    #     json.dump(scores, outfile, indent=4)
