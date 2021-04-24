import json
from utils import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data_path = "../data/"
    domains = ["tw", "az", "mv"]

    # Load data.

    bert_dict = load_bert(data_path, domains, 2000)
    y_dict = load_np_files(data_path, domains, ['train', 'dev'], load_feature=False)
    for b in bert_dict:
        print(b, len(bert_dict[b]))
    for y in y_dict:
        print(y, len(y_dict[y]))

    # Last Model.
    lr = LogisticRegression(C=0.1, max_iter=200000)
    final_accuracies_all_domains = {}
    accuracies_ti_all_domains = {}
    dists_all_domains = {}
    data_size = 2000

    for source in domains:
        for target in domains:
            train_name = source + "_to_" + target
            print("\n", train_name)
            source_feature_name = source
            source_label_name = "y_train_" + source
            target_feature_name = target
            target_label_name = "y_train_" + target

            final_accuracies, accuracies_ti, dists = gradual_train_groups_range(
                x_source_raw=bert_dict[source_feature_name], y_source_raw=y_dict[source_label_name],
                x_target_raw=bert_dict[target_feature_name], y_target_raw=y_dict[target_label_name],
                base_model=lr, data_size=data_size, group_range=[0, 10], plot_hist=False, dist_type="l2",
                conf=0.3
            )

            final_accuracies_all_domains[train_name] = final_accuracies
            accuracies_ti_all_domains[train_name] = accuracies_ti
            dists_all_domains[train_name] = dists

    print(accuracies_ti_all_domains)
    with open("../outputs/accuracies_ti_all_domains_conf3_lr_c0.1_l2.json", "w") as outfile:
        json.dump(accuracies_ti_all_domains, outfile, indent=4)

    # # Self-train label as final prediction label.
    # lr = LogisticRegression(C=0.1, max_iter=200000)
    # accuracies_all_domains = {}
    # data_size = 2000
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
    #         accuracies_all_domains[train_name] = gradual_train_conf_groups(
    #             x_source_raw=bert_dict[source_feature_name], y_source_raw=y_dict[source_label_name],
    #             x_target_raw=bert_dict[target_feature_name], y_target_raw=y_dict[target_label_name],
    #             base_model=lr, data_size=data_size,  conf=0.9
    #         )
    #
    # print(accuracies_all_domains)
    # with open("../outputs/accuracies_all_domains_s2t_conf9.json", "w") as outfile:
    #     json.dump(accuracies_all_domains, outfile, indent=4)

    # # S2T_pr_adj_blc.
    # lr = LogisticRegression(C=0.1, max_iter=200000)
    # accuracies_all_domains = {}
    # data_size = 2000
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
    #         accuracies_all_domains[train_name] = S2T_p4_adj_blc(
    #             train_features=bert_dict[source_feature_name], train_labels=y_dict[source_label_name][:data_size],
    #             test_features=bert_dict[target_feature_name], test_labels=y_dict[target_label_name][:data_size],
    #             base_model=lr, dist_eval=False
    #         )
    # print(accuracies_all_domains)
    # with open("../outputs/accuracies_all_domains_S2T_p4_adj_blc_lr_c0.1.json", "w") as outfile:
    #     json.dump(accuracies_all_domains, outfile, indent=4)
