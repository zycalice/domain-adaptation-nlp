import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Helper functions.

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def load_np_files(data_path, domains, data_types, load_feature):
    np_dataset = {}
    for domain in domains:
        for data_type in data_types:
            if load_feature:
                file_prefix = "X_"
            else:
                file_prefix = "y_"
            file = file_prefix + data_type + "_" + domain
            filename = data_path + "all_cleaned/" + file + ".npy"
            np_dataset[file] = np.load(filename, allow_pickle=True)
    return np_dataset


def load_bert(data_path, domains, data_size):
    bert_dataset = {}
    for domain in domains:
        bert_dataset[domain] = np.load(data_path + "all_bert/encoded_" +
                                       domain + "_train_" + str(data_size) + ".npy")
    return bert_dataset


# Pseudo labeling (self train) and gradual train.

def psuedo_labeling_lm(x_source, y_source, x_ti, y_ti, model, conf=0):
    # TODO: add NER version; NER version inputs are dictionaries
    base_model = model
    model.fit(x_source, y_source)
    y_prob = base_model.predict_proba(x_ti)[:, 0]
    X_ti_keep = x_ti[(y_prob >= 0.5 + conf) | (y_prob < 0.5 - conf)]
    y_pred = model.predict(X_ti_keep)
    X_source_updated = np.concatenate((x_source, X_ti_keep), 0)
    y_source_updated = np.concatenate((y_source, y_pred), 0)
    return X_source_updated, y_source_updated


def gradual_train_dist_groups(x_source, y_source, x_target, y_target, base_model, dists, group_size=5, conf=0):
    # initiate model
    model = base_model

    # create groups within targets and gradually train
    dists = np.array(dists)
    dists_order = np.argsort(dists)
    dists_rank = np.argsort(dists_order)

    step = len(y_target) / group_size
    x_target_groups = []
    y_target_groups = []
    gradual_scores = []
    x_source_updated = x_source.copy()
    y_source_updated = y_source.copy()
    for i in range(group_size):
        subset_tf = (step * i <= dists_rank) & (dists_rank < step * (i + 1))
        X_ti = x_target[subset_tf]
        y_ti = y_target[subset_tf]
        x_target_groups.append(X_ti)
        y_target_groups.append(y_ti)
        x_source_updated, y_source_updated = psuedo_labeling_lm(x_source_updated, y_source_updated, X_ti, y_ti, model,
                                                                conf)
        gradual_score = model.fit(x_source_updated, y_source_updated).score(x_target, y_target)
        gradual_scores.append(gradual_score)
    return gradual_scores


def gradual_train_groups_range(x_source_raw, y_source_raw, x_target_raw, y_target_raw, base_model, data_size,
                               group_range, conf, plot_hist=True):
    # initial data and model
    data_size = min(len(x_source_raw), len(x_target_raw), data_size)
    print(data_size)
    x_source, y_source = x_source_raw[:data_size], y_source_raw[:data_size]
    x_target, y_target = x_target_raw[:data_size], y_target_raw[:data_size]
    print(x_source.shape, y_source.shape, x_target.shape, y_target.shape)

    model = base_model
    model.fit(x_source, y_source)
    no_self_train_adaptation_score = model.score(x_target, y_target)

    # calculate distances
    source_center = np.mean(x_source, 0)
    dists = [1 - cosine_similarity(source_center.reshape(1, -1), x.reshape(1, -1))[0][0] for x in x_target]
    if plot_hist:
        plt.hist(dists, bins=100)
        plt.show()

    # save accuracies
    final_accuracies = [no_self_train_adaptation_score]
    accuracies_ti = {"no_self_train_adaptation_score": no_self_train_adaptation_score}
    for i in range(group_range[0] + 1, group_range[1] + 1):
        data_size = data_size
        base_model = base_model
        gradual_scores = gradual_train_dist_groups(
            x_source, y_source,
            x_target, y_target,
            base_model, dists=dists, group_size=i, conf=conf)
        final_accuracies.append(gradual_scores[-1])
        accuracies_ti[i] = gradual_scores
        print("group", i, '{:.2f}'.format(no_self_train_adaptation_score * 100),
              ['{:.2f}'.format(elem * 100) for elem in gradual_scores])

    return final_accuracies, accuracies_ti, dists


# Self-Train and use pseudo level as final labels, and use conf as groups.

def psuedo_labeling_label_final(x_s, y_s, x_t, y_t, model, conf):
    # TODO: add NER version; NER version inputs are dictionaries
    base_model = model
    model.fit(x_s, y_s)
    y_prob = base_model.predict_proba(x_t)[:, 0]
    x_ti_keep = []
    x_ti_not_keep = []
    y_ti_keep = []
    y_ti_not_keep = []

    # if no data past the conf requirement, lower the requirement by 0.1
    while len(x_ti_keep) == 0:
        keep_ti_bool = (y_prob >= 0.5 + conf) | (y_prob < 0.5 - conf)
        x_ti_keep = x_t[keep_ti_bool]
        x_ti_not_keep = x_t[~keep_ti_bool]
        y_ti_keep = y_t[keep_ti_bool]
        y_ti_not_keep = y_t[~keep_ti_bool]
        conf = conf - 0.01

    # output prediction and update source
    y_pred = model.predict(x_ti_keep)
    x_source_updated = np.concatenate((x_s, x_ti_keep), 0)
    y_source_updated = np.concatenate((y_s, y_pred), 0)
    # print(len(x_ti_not_keep))
    return x_source_updated, y_source_updated, y_pred, y_ti_keep, x_ti_not_keep, y_ti_not_keep


def gradual_train_conf_groups(x_source_raw, y_source_raw, x_target_raw, y_target_raw, base_model, data_size, conf):
    # initiate values
    data_size = min(len(x_source_raw), len(x_target_raw), data_size)
    print(data_size)
    x_source_raw, y_source_raw = x_source_raw[:data_size], y_source_raw[:data_size]
    x_target_raw, y_target_raw = x_target_raw[:data_size], y_target_raw[:data_size]
    print(x_source_raw.shape, y_source_raw.shape, x_target_raw.shape, y_target_raw.shape)

    x_source, y_source = x_source_raw.copy(), y_source_raw.copy()
    x_target, y_target = x_target_raw.copy(), y_target_raw.copy()
    y_pred_all = []
    y_true_all = []

    # repeat self-train until all target data are computed
    while len(x_target) > 0:
        x_source, y_source, y_pred, y_true, x_target, y_target = psuedo_labeling_label_final(
            x_source, y_source, x_target, y_target, base_model, conf
        )
        y_pred_all.extend(list(y_pred))
        y_true_all.extend(list(y_true))

    # calculate accuracy
    s2t_score = base_model.fit(x_source_raw, y_source_raw).score(x_target_raw, y_target_raw)
    s2s_score = base_model.fit(x_source_raw, y_source_raw).score(x_source_raw, y_source_raw)
    t2t_score = base_model.fit(x_target_raw, y_target_raw).score(x_target_raw, y_target_raw)
    gradual_score = accuracy_score(y_true_all, y_pred_all)
    return s2s_score, t2t_score, s2t_score, gradual_score


#######################################################################################################################
def cos_dist(A, B):
    return 1 - (np.dot(A, B) / (np.norm(A) * np.norm(B)))


def S2T(train_features, train_labels, test_features, test_labels):
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    return lr_clf.score(test_features, test_labels)
