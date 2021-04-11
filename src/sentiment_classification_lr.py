import json

from utils import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data_path = "../data/"
    domains = ["tw", "az", "mv", "fi"]

    # Load data
    bert_dict = load_bert(data_path, domains, 2000)
    y_dict = load_np_files(data_path, domains, ['train', 'dev'], load_feature=False)
    for b in bert_dict:
        print(b, len(bert_dict[b]))
    for y in y_dict:
        print(y, len(y_dict[y]))

    # # test
    # lr = LogisticRegression(max_iter=200000)
    # final_accuracies, accuracies_ti, dists = gradual_train_groups(
    #     X_source_raw=bert_dict["az2000"], y_source_raw=y_dict["y_train_az"],
    #     X_target_raw=bert_dict["fi2000"], y_target_raw=y_dict["y_train_fi"],
    #     base_model=lr, data_size=data_size, group_range=[0, 20], plot_hist=False,
    # )

    # Create accuracies and graphs
    lr = LogisticRegression(max_iter=200000)
    final_accuracies_all_domains = {}
    accuracies_ti_all_domains = {}
    dists_all_domains = {}

    for source in domains:
        for target in domains:
            train_name = source + "_to_" + target
            print("\n", train_name)
            source_feature_name = source
            source_label_name = "y_train_" + source
            target_feature_name = target
            target_label_name = "y_train_" + target

            if (source == "fi") or (target == "fi"):
                data_size = 1185
            else:
                data_size = 2000

            final_accuracies, accuracies_ti, dists = gradual_train_groups(
                X_source_raw=bert_dict[source_feature_name], y_source_raw=y_dict[source_label_name],
                X_target_raw=bert_dict[target_feature_name], y_target_raw=y_dict[target_label_name],
                base_model=lr, data_size=data_size, group_range=[0, 10], plot_hist=False,
            )

            final_accuracies_all_domains[train_name] = final_accuracies
            accuracies_ti_all_domains[train_name] = accuracies_ti
            dists_all_domains[train_name] = dists

    print(accuracies_ti_all_domains)
    with open("../outputs/accuracies_ti_all_domains.json", "w") as outfile:
        json.dump(accuracies_ti_all_domains, outfile, indent=4)
