import json
from utils import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data_path = "../data/"
    domains = ["tw", "az", "mv", "fi"]

    # Load data.

    bert_dict = load_bert(data_path, domains, 2000)
    y_dict = load_np_files(data_path, domains, ['train', 'dev'], load_feature=False)
    for b in bert_dict:
        print(b, len(bert_dict[b]))
    for y in y_dict:
        print(y, len(y_dict[y]))

    # Last Model.

    # # Create accuracies and graphs
    # lr = LogisticRegression(max_iter=200000)
    # final_accuracies_all_domains = {}
    # accuracies_ti_all_domains = {}
    # dists_all_domains = {}
    #
    # for source in domains:
    #     for target in domains:
    #         train_name = source + "_to_" + target
    #         print("\n", train_name)
    #         source_feature_name = source
    #         source_label_name = "y_train_" + source
    #         target_feature_name = target
    #         target_label_name = "y_train_" + target
    #
    #         if (source == "fi") or (target == "fi"):
    #             data_size = 1185
    #         else:
    #             data_size = 2000
    #
    #         final_accuracies, accuracies_ti, dists = gradual_train_groups_range(
    #             x_source_raw=bert_dict[source_feature_name], y_source_raw=y_dict[source_label_name],
    #             x_target_raw=bert_dict[target_feature_name], y_target_raw=y_dict[target_label_name],
    #             base_model=lr, data_size=data_size, group_range=[0, 10], plot_hist=False,
    #             conf=0.3
    #         )
    #
    #         final_accuracies_all_domains[train_name] = final_accuracies
    #         accuracies_ti_all_domains[train_name] = accuracies_ti
    #         dists_all_domains[train_name] = dists
    #
    # print(accuracies_ti_all_domains)
    # with open("../outputs/accuracies_ti_all_domains_conf3.json", "w") as outfile:
    #     json.dump(accuracies_ti_all_domains, outfile, indent=4)

    # Self-train label as final prediction label
    lr = LogisticRegression(max_iter=200000)
    accuracies_all_domains = {}
    data_size = 2000

    for source in domains:
        for target in domains:
            train_name = source + "_to_" + target
            print("\n", train_name)
            source_feature_name = source
            source_label_name = "y_train_" + source
            target_feature_name = target
            target_label_name = "y_train_" + target

            accuracies_all_domains[train_name] = gradual_train_conf_groups(
                x_source_raw=bert_dict[source_feature_name], y_source_raw=y_dict[source_label_name],
                x_target_raw=bert_dict[target_feature_name], y_target_raw=y_dict[target_label_name],
                base_model=lr, data_size=data_size,  conf=0.4
            )

    print(accuracies_all_domains)
    with open("../outputs/accuracies_all_domains_s2t_conf4.json", "w") as outfile:
        json.dump(accuracies_all_domains, outfile, indent=4)

    # accuracies_all_domains = {}
    # lr = LogisticRegression(max_iter=200000)
    #
    # for source in domains:
    #     for target in domains:
    #         train_name = source + "_to_" + target
    #         print("\n", train_name)
    #         source_feature_name = source
    #         source_label_name = "y_train_" + source
    #         target_feature_name = target
    #         target_label_name = "y_train_" + target
    #
    #         data_size = min(len(bert_dict[source_feature_name]), len(bert_dict[target_feature_name]))
    #         print(data_size)
    #         try:
    #             accuracies_all_domains[train_name] = S2T_prob_4_adj(
    #                 train_features=bert_dict[source_feature_name][:data_size],
    #                 train_labels=y_dict[source_label_name][:data_size],
    #                 test_features=bert_dict[target_feature_name][:data_size],
    #                 test_labels=y_dict[target_label_name][:data_size],
    #                 base_model=lr,
    #             )
    #
    #         except:
    #             pass
    #
    # with open("../outputs/accuracies_all_domains.json", "w") as outfile:
    #     json.dump(accuracies_all_domains, outfile, indent=4)
