import pickle
import json
from utils import *
from sklearn.linear_model import LogisticRegression


def run_check_version(few_shot, output_path):
    # initiate.
    with open("../data/amazon_reviews/amazon_4.pickle", "rb") as fr:
        all_data = pickle.load(fr)

    lr = LogisticRegression(C=0.1, max_iter=200000)
    accuracies_all_domains = {}
    data_size = 2000

    for source in all_data:
        for target in all_data:
            train_name = source[2] + "_to_" + target[2]
            print("\n", train_name)

            accuracies_all_domains[train_name] = run_gradual_train_balanced_conf_groups(
                x_source_raw=source[0], y_source_raw=source[1],
                x_target_raw=target[0], y_target_raw=target[1],
                base_model=lr, data_size=data_size, top_n=100,
                few_shot=few_shot,
            )

    print(accuracies_all_domains)
    with open(output_path, "w") as outfile:
        json.dump(accuracies_all_domains, outfile, indent=4)


def run_main_version():
    # initiate.
    with open("../data/amazon_reviews/amazon_4.pickle", "rb") as fr:
        all_data = pickle.load(fr)

    lr = LogisticRegression(C=0.1, max_iter=200000)
    accuracies_all_domains = {}
    data_size = 2000

    for source in all_data:
        for target in all_data:
            train_name = source[2] + "_to_" + target[2]
            print("\n", train_name)

            accuracies_all_domains[train_name] = S2T_p4_adj_blc(
                train_features=source[0], train_labels=source[1],
                test_features=target[0], test_labels=target[1],
                model=lr, num_i=100,
            )

    print(accuracies_all_domains)
    with open("../outputs/accuracies_ti_amazon_S2T_p4_adj_blc_c0.1.json", "w") as outfile:
        json.dump(accuracies_all_domains, outfile, indent=4)


if __name__ == '__main__':
    pass

    run_check_version(few_shot=None,
                      output_path="../outputs/accuracies_ti_amazon_conf_blc_c0.1.json")
