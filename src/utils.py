import random
import numpy as np
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


# Distances.

def cosine_dist(x_source, x_target):
    source_center = np.mean(x_source, 0)
    dists = [1 - cosine_similarity(source_center.reshape(1, -1), x.reshape(1, -1))[0][0] for x in x_target]
    return dists


def l2_dist(x_source, x_target):
    source_center = np.mean(x_source, 0)
    dists = [np.linalg.norm(source_center.reshape(1, -1) - x.reshape(1, -1)) for x in x_target]
    return dists


def get_dist(x_source, x_target, dist_type):
    """
    get dist function
    :param x_source:
    :param x_target:
    :param dist_type: cosine similarity, l2 norm,  maximum mean discrepancy, fisher
    :return:
    """
    if dist_type not in ["cos", "l2"]:
        raise ValueError("")

    dists = None
    if dist_type == "cos":
        dists = cosine_dist(x_source, x_target)
    if dist_type == "l2":
        dists = l2_dist(x_source, x_target)

    return dists


# Pseudo labeling (self train) and gradual train.

def pseudo_labeling(x_source, y_source, x_ti, y_ti, model, conf=0, few_shot_size=0, label_final=False):
    # TODO: add NER version; NER version inputs are dictionaries
    base_model = model

    if few_shot_size != 0:
        idx = np.arange(len(y_ti))
        # print(len(x_ti), type(x_ti), x_ti.shape)
        selected_idx = np.array(random.sample(list(idx), max(1, int(few_shot_size * len(y_ti)))))
        # print(selected_idx)
        selected_label = y_ti[selected_idx]
        selected_features = x_ti[selected_idx]
        x_source = np.concatenate((x_source, selected_features), 0)
        y_source = np.concatenate((y_source, selected_label), 0)
        x_ti = np.delete(np.array(x_ti), selected_idx, 0)
        y_ti = np.delete(np.array(y_ti), selected_idx, 0)
        # print(len(x_ti), len(y_ti))

    model.fit(x_source, y_source)

    if label_final:
        y_pred = model.predict(x_ti)
        x_source_updated = np.concatenate((x_source, x_ti), 0)
        y_source_updated = np.concatenate((y_source, y_pred), 0)
        return x_source_updated, y_source_updated, y_pred

    else:
        y_prob = base_model.predict_proba(x_ti)[:, 0]
        x_ti_keep = x_ti[(y_prob >= 0.5 + conf) | (y_prob < 0.5 - conf)]
        if len(x_ti_keep) == 0:
            x_ti_keep = x_ti
        y_pred = model.predict(x_ti_keep)
        x_source_updated = np.concatenate((x_source, x_ti_keep), 0)
        y_source_updated = np.concatenate((y_source, y_pred), 0)
        return x_source_updated, y_source_updated, None, x_ti, y_ti


def gradual_train_dist_groups(x_source, y_source, x_target, y_target, base_model, dists,
                              group_size=5, conf=0, subset_size=None, few_shot_size=0,
                              label_final=False):
    """
    For a particular group size
    When subset size activates, there is only one group
    """
    # initiate model
    model = base_model
    x_source_updated = x_source.copy()
    y_source_updated = y_source.copy()

    # calculate distance ranks
    dists = np.array(dists)
    dists_order = np.argsort(dists)
    dists_rank = np.argsort(dists_order)
    step = len(y_target) / group_size

    if (subset_size is not None) and (group_size == 1):
        # generate subset results
        subset_tf = dists_rank <= len(y_source) * subset_size
        x_ti = x_target[subset_tf]
        y_ti = y_target[subset_tf]
        x_source_updated, y_source_updated, _ = pseudo_labeling(x_source_updated, y_source_updated, x_ti, y_ti,
                                                                model, conf, few_shot_size)
        subset_score = model.fit(x_source_updated, y_source_updated).score(x_target, y_target)
        return subset_score

    if (subset_size is None) and (group_size > 0):
        # create groups within targets and gradually train
        x_target_groups = []
        y_target_groups = []
        gradual_scores = []
        y_preds = []
        y_ordered_target = []
        x_t_new = []
        y_t_new = []

        for i in range(group_size):
            subset_tf = (dists_rank >= step * i) & (dists_rank < step * (i + 1))
            x_ti = x_target[subset_tf]
            y_ti = y_target[subset_tf]
            x_target_groups.append(x_ti)
            y_target_groups.append(y_ti)
            x_source_updated, y_source_updated, y_pred, x_t, y_t = pseudo_labeling(x_source_updated, y_source_updated,
                                                                                   x_ti, y_ti,
                                                                                   model, conf, few_shot_size,
                                                                                   label_final)
            if label_final:
                y_preds.extend(y_pred)
                y_ordered_target.extend(y_ti)
            else:
                x_t_new.extend(x_t)
                y_t_new.extend(y_t)
                # print(len(x_t), len(y_t))
                gradual_score = model.fit(x_source_updated, y_source_updated).score(x_t_new, y_t_new)
                gradual_scores.append(gradual_score)

        if label_final:
            model.fit(x_source_updated, y_source_updated)
            # print(len(y_preds))
            gradual_score = [accuracy_score(y_preds, y_ordered_target)]
            return gradual_score
        else:
            return gradual_scores

    raise ValueError("Check input; one of group size or subset size should be none.")


def run_gradual_train_ranges(x_source_raw, y_source_raw, x_target_raw, y_target_raw, base_model, data_size,
                             group_range, subset_range, conf, dist_type, plot_hist=True, few_shot_size=0,
                             label_final=False):
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
    dists = get_dist(x_source, x_target, dist_type)
    if plot_hist:
        plt.hist(dists, bins=100)
        plt.show()

    # save accuracies
    final_accuracies = [no_self_train_adaptation_score]
    accuracies_ti = {"no_self_train_adaptation_score": no_self_train_adaptation_score}
    base_model = base_model

    # subset version
    if (subset_range is not None) and (group_range is None):
        for x in subset_range:
            subset_score = gradual_train_dist_groups(
                x_source, y_source,
                x_target, y_target,
                base_model=base_model, dists=dists, group_size=1, subset_size=x, conf=conf,
                few_shot_size=few_shot_size,
            )
            final_accuracies.append(subset_score)
            accuracies_ti[x] = subset_score
            print('subset', '{:.0f}%'.format(x * 100), '{:.2f}%'.format(no_self_train_adaptation_score * 100),
                  '{:.2f}%'.format(subset_score * 100))
        return final_accuracies, accuracies_ti, dists

    # group train version
    if (subset_range is None) and (group_range is not None):
        for i in group_range:
            gradual_scores = gradual_train_dist_groups(
                x_source, y_source,
                x_target, y_target,
                base_model=base_model, dists=dists, group_size=i, subset_size=None, conf=conf,
                few_shot_size=few_shot_size, label_final=label_final
            )
            final_accuracies.append(gradual_scores[-1])
            accuracies_ti[i] = gradual_scores
            print('group', i, '{:.2f}'.format(no_self_train_adaptation_score * 100),
                  ['{:.2f}'.format(elem * 100) for elem in gradual_scores])

        return final_accuracies, accuracies_ti, dists

    raise ValueError("Check inputs for group range and subset range. Only one should be not none.")


# Self-Train and use pseudo level as final labels, and use conf as groups.

def pseudo_labeling_label_final_conf(x_source, y_source, x_ti, y_ti, model, conf, few_shot_size=0):
    # TODO: add NER version; NER version inputs are dictionaries
    base_model = model
    model.fit(x_source, y_source)
    y_prob = base_model.predict_proba(x_ti)[:, 0]
    x_ti_keep = []
    x_ti_not_keep = []
    y_ti_keep = []
    y_ti_not_keep = []

    if few_shot_size != 0:
        idx = np.arange(len(y_ti))
        selected_idx = np.array(random.sample(list(idx), max(1, int(few_shot_size * len(y_ti)))))
        selected_label = y_ti[selected_idx]
        selected_features = x_ti[selected_idx]
        x_source = np.concatenate((x_source, selected_features), 0)
        y_source = np.concatenate((y_source, selected_label), 0)
        x_ti = np.delete(np.array(x_ti), selected_idx, 0)
        y_ti = np.delete(np.array(y_ti), selected_idx, 0)

    # if no data past the conf requirement, lower the requirement by 0.1
    while len(x_ti_keep) == 0:
        keep_ti_bool = (y_prob >= 0.5 + conf) | (y_prob < 0.5 - conf)
        x_ti_keep = x_ti[keep_ti_bool]
        x_ti_not_keep = x_ti[~keep_ti_bool]
        y_ti_keep = y_ti[keep_ti_bool]
        y_ti_not_keep = y_ti[~keep_ti_bool]
        conf = conf - 0.01

    # output prediction and update source
    y_pred = model.predict(x_ti_keep)
    x_source_updated = np.concatenate((x_source, x_ti_keep), 0)
    y_source_updated = np.concatenate((y_source, y_pred), 0)
    return x_source_updated, y_source_updated, y_pred, y_ti_keep, x_ti_not_keep, y_ti_not_keep


def run_gradual_train_final_label_conf_groups(x_source_raw, y_source_raw, x_target_raw, y_target_raw, base_model,
                                              data_size, conf,
                                              few_shot_size=0):
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
        x_source, y_source, y_pred, y_true, x_target, y_target = pseudo_labeling_label_final_conf(
            x_source, y_source, x_target, y_target, base_model, conf, few_shot_size
        )
        y_pred_all.extend(list(y_pred))
        y_true_all.extend(list(y_true))

    # calculate accuracy
    s2t_score = base_model.fit(x_source_raw, y_source_raw).score(x_target_raw, y_target_raw)
    s2s_score = base_model.fit(x_source_raw, y_source_raw).score(x_source_raw, y_source_raw)
    t2t_score = base_model.fit(x_target_raw, y_target_raw).score(x_target_raw, y_target_raw)
    gradual_score = accuracy_score(y_true_all, y_pred_all)
    return s2s_score, t2t_score, s2t_score, gradual_score

##################################################################################################################
# Used in actual report.


def self_train(x_source, y_source, x_ti, y_ti, base_model):
    base_model.fit(x_source, y_source)
    y_pred_ti = base_model.predict(x_ti)
    base_model.fit(np.concatenate((x_source, x_ti), 0), np.concatenate((y_source, y_pred_ti), 0))
    self_train_score = base_model.score(x_ti, y_ti)
    return self_train_score


def fs_train(x_source, y_source, x_ti, y_ti, indexes, base_model):
    x_source_fs_least = np.concatenate((x_source, x_ti[indexes]), 0)
    y_source_fs_least = np.concatenate((y_source, y_ti[indexes]), 0)
    x_target_fs_least = np.delete(x_ti, [indexes], 0)
    y_target_fs_least = np.delete(y_ti, [indexes], 0)
    print(x_source_fs_least.shape, y_source_fs_least.shape, x_target_fs_least.shape, y_target_fs_least.shape)
    score = base_model.fit(x_source_fs_least, y_source_fs_least).score(x_target_fs_least, y_target_fs_least)
    return score


# Balanced conf.
def pseudo_label_balanced(x_source, y_source, x_ti, y_ti, base_model, top_n,
                          use_dist, few_shot=None):
    # get predictions
    base_model.fit(x_source, y_source)
    y_prob_ti = base_model.predict_proba(x_ti)[:, 0]
    y_pred_ti = base_model.predict(x_ti)

    # add dist
    dist = get_dist(x_source, x_ti, "cos")

    # change to arrays
    x_source = np.array(x_source)
    y_source = np.array(y_source)
    x_ti = np.array(x_ti)

    # group variables
    if use_dist:
        targets = [(i, dist, x_ti[i], y_ti[i], y_pred_ti[i]) for i, dist in enumerate(dist)]
        sorted_targets = sorted(targets, key=lambda x: x[1])
        index = np.arange(len(targets))
        pos_idx = index[y_prob_ti >= 0.5]
        neg_idx = index[y_prob_ti < 0.5]
        targets_pos = [t for t in targets if t[0] in pos_idx]
        targets_neg = [t for t in targets if t[0] in neg_idx]
        sorted_targets_pos = sorted(targets_pos, key=lambda x: x[1])
        sorted_targets_neg = sorted(targets_neg, key=lambda x: x[1])
        keep_n = min(sum(y_prob_ti < 0.5), sum(y_prob_ti >= 0.5), top_n)
        print(keep_n, len(targets_neg), len(targets_pos))
        targets_keep = sorted_targets_pos[:keep_n] + sorted_targets_neg[:keep_n]  # top 100 and bottom 100
        targets_left = sorted_targets_pos[keep_n:] + sorted_targets_neg[keep_n:]
        print(len(targets_keep))
        fs_idx = None
    else:
        targets = [(i, prob, x_ti[i], y_ti[i], y_pred_ti[i]) for i, prob in enumerate(y_prob_ti)]
        sorted_targets = sorted(targets, key=lambda x: x[1])
        keep_n = min(sum(y_prob_ti < 0.5), sum(y_prob_ti >= 0.5), top_n)
        targets_keep = sorted_targets[:keep_n] + sorted_targets[-keep_n:]  # top 100 and bottom 100
        targets_left = sorted_targets[keep_n:-keep_n]
        fs_idx = None

    # update
    if few_shot is not None:
        if few_shot == "random":
            keep_idx = [t[0] for t in targets_keep]
            fs_idx = random.choice(keep_idx)

        if few_shot == "least":
            if use_dist:
                fs_idx = targets_keep[0]
            else:
                least_neg = sorted_targets[0]
                least_pos = sorted_targets[-1]

                if abs(least_neg[4] - 0.5) < abs(least_pos[4] - 0.5):
                    fs_idx = least_neg[0]
                else:
                    fs_idx = least_pos[0]

        targets_keep = [t for t in targets_keep if t[0] != fs_idx]
        print(x_source.shape, y_source.shape)
        x_source = np.concatenate((x_source, x_ti[fs_idx].reshape(1, -1)), 0)
        print(x_source.shape, y_source.shape)
        y_source = np.concatenate((y_source, [y_ti[fs_idx]]), 0)

    if keep_n != 0:
        x_source_updated = np.concatenate((x_source, np.array([t[2] for t in targets_keep])), 0)
        y_source_updated = np.concatenate((y_source, np.array([t[4] for t in targets_keep])), 0)
        x_ti_left = [t[2] for t in targets_left]
        y_ti_left = [t[3] for t in targets_left]
    else:
        x_source_updated = x_source
        y_source_updated = y_source
        x_ti_left = []
        y_ti_left = []

    return x_source_updated, y_source_updated, x_ti_left, y_ti_left, fs_idx


def pseudo_label_unbalanced(x_source, y_source, x_ti, y_ti, base_model, top_n,
                            use_dist, few_shot=None):
    # get predictions
    base_model.fit(x_source, y_source)
    y_prob_ti = base_model.predict_proba(x_ti)[:, 0]
    y_pred_ti = base_model.predict(x_ti)

    # add dist
    dist = get_dist(x_source, x_ti, "cos")

    # change to arrays
    x_source = np.array(x_source)
    y_source = np.array(y_source)
    x_ti = np.array(x_ti)

    # group variables
    if use_dist:
        targets = [(i, dist, x_ti[i], y_ti[i], y_pred_ti[i]) for i, dist in enumerate(dist)]
        sorted_targets = sorted(targets, key=lambda x: x[1])
        keep_n = min(len(y_prob_ti), 2 * top_n)
        targets_keep = sorted_targets[:keep_n]  # top 200
        targets_left = sorted_targets[keep_n:]
        fs_idx = None
    else:
        targets = [(i, abs(prob - 0.5), x_ti[i], y_ti[i], y_pred_ti[i]) for i, prob in enumerate(y_prob_ti)]
        sorted_targets = sorted(targets, key=lambda x: x[1], reverse=True)
        keep_n = min(len(y_prob_ti), 2 * top_n)
        targets_keep = sorted_targets[:keep_n]  # largest 200
        targets_left = sorted_targets[keep_n:]
        fs_idx = None

    # update
    if few_shot is not None:
        if few_shot == "random":
            keep_idx = [t[0] for t in targets_keep]
            fs_idx = random.choice(keep_idx)

        if few_shot == "least":
            if use_dist:
                fs_idx = targets_keep[0]
            else:
                least_neg = sorted_targets[0]
                least_pos = sorted_targets[-1]

                if abs(least_neg[4] - 0.5) < abs(least_pos[4] - 0.5):
                    fs_idx = least_neg[0]
                else:
                    fs_idx = least_pos[0]

        targets_keep = [t for t in targets_keep if t[0] != fs_idx]
        print(x_source.shape, y_source.shape)
        x_source = np.concatenate((x_source, x_ti[fs_idx].reshape(1, -1)), 0)
        print(x_source.shape, y_source.shape)
        y_source = np.concatenate((y_source, [y_ti[fs_idx]]), 0)

    if keep_n != 0:
        x_source_updated = np.concatenate((x_source, np.array([t[2] for t in targets_keep])), 0)
        y_source_updated = np.concatenate((y_source, np.array([t[4] for t in targets_keep])), 0)
        x_ti_left = [t[2] for t in targets_left]
        y_ti_left = [t[3] for t in targets_left]
    else:
        x_source_updated = x_source
        y_source_updated = y_source
        x_ti_left = []
        y_ti_left = []

    return x_source_updated, y_source_updated, x_ti_left, y_ti_left, fs_idx


def run_gradual_train_groups(x_source_raw, y_source_raw, x_target_raw, y_target_raw, base_model,
                             data_size, top_n, balanced=True, use_dist=False, few_shot=None):
    # initiate values
    data_size = min(len(x_source_raw), len(x_target_raw), data_size)
    print(data_size)
    x_source_raw, y_source_raw = x_source_raw[:data_size], y_source_raw[:data_size]
    x_target_raw, y_target_raw = x_target_raw[:data_size], y_target_raw[:data_size]
    print(x_source_raw.shape, y_source_raw.shape, x_target_raw.shape, y_target_raw.shape)

    x_source, y_source = x_source_raw.copy(), y_source_raw.copy()
    x_target, y_target = x_target_raw.copy(), y_target_raw.copy()

    # repeat self-train until all target data are computed
    num_iter = 0
    fs_indexes = []
    while len(x_target) > 0:
        if balanced:
            x_source, y_source, x_target, y_target, fs_idx = pseudo_label_balanced(
                x_source, y_source, x_target, y_target, base_model, top_n, use_dist, few_shot
            )
        else:
            x_source, y_source, x_target, y_target, fs_idx = pseudo_label_unbalanced(
                x_source, y_source, x_target, y_target, base_model, top_n, use_dist, few_shot
            )
        num_iter += 1
        fs_indexes.append(fs_idx)
        print("Total target len after this iteration:", len(x_target))

    # calculate accuracy
    if few_shot is None:
        s2t_score = base_model.fit(x_source_raw, y_source_raw).score(x_target_raw, y_target_raw)
        s2s_score = base_model.fit(x_source_raw, y_source_raw).score(x_source_raw, y_source_raw)
        t2t_score = base_model.fit(x_target_raw, y_target_raw).score(x_target_raw, y_target_raw)
        self_train_score = self_train(x_source_raw, y_source_raw, x_target_raw, y_target_raw, base_model)
        gradual_score = base_model.fit(x_source, y_source).score(x_target_raw, y_target_raw)
        print(s2s_score, t2t_score, s2t_score, gradual_score)

        all_scores = {
            's2s': s2s_score,
            't2t': t2t_score,
            's2t': s2t_score,
            'self_train': self_train_score,
            'gradual_self_train': gradual_score,
            'num_iter': num_iter
        }

    else:
        # fs using the same index
        fs_same_idx_sc = fs_train(x_source_raw, y_source_raw, x_target_raw, y_target_raw, fs_indexes, base_model)

        # fs using random labels
        random_indexes = random.sample(list(np.arange(len(y_target_raw))), len(fs_indexes))
        fs_random_idx_sc = fs_train(x_source_raw, y_source_raw, x_target_raw, y_target_raw, random_indexes, base_model)

        # fs using the least confident labels
        y_prob_abs = abs(base_model.fit(x_source_raw, y_source_raw).predict_proba(x_target_raw)[:, 0] - 0.5)
        y_idx_prob_abs = [(i, e) for i, e in enumerate(y_prob_abs)]
        y_idx_prob_abs = sorted(y_idx_prob_abs, key=lambda x: x[1])
        least_indexes = [t[0] for t in y_idx_prob_abs[:len(fs_indexes)]]
        fs_least_idx_sc = fs_train(x_source_raw, y_source_raw, x_target_raw, y_target_raw, least_indexes, base_model)

        # gradual train score
        x_target_fs_least = np.delete(x_target_raw, [fs_indexes], 0)
        y_target_fs_least = np.delete(y_target_raw, [fs_indexes], 0)
        gradual_score = base_model.fit(x_source, y_source).score(x_target_fs_least, y_target_fs_least)

        all_scores = {
            "fs_same_index": fs_same_idx_sc,
            "fs_random_index": fs_random_idx_sc,
            "fs_least_index": fs_least_idx_sc,
            "fs_gradual_self_train": gradual_score,
            "num_iter": num_iter,
        }

    return all_scores


#######################################################################################################################
def cos_dist(A, B):
    return 1 - (np.dot(A, B) / (np.norm(A) * np.norm(B)))


def S2T_p4_adj_blc(train_features, train_labels, test_features, test_labels, model, num_i):
    top_n = num_i
    original_score = model.fit(train_features, train_labels).score(test_features, test_labels)

    # gradual training
    X_train = train_features[:]
    y_train = train_labels[:]
    X_test = test_features[:]
    y_test = test_labels[:]
    y_pred_store = []
    y_test_store = []
    previous_r_target = []

    while len(X_test) > 0:
        lr_clf = model
        lr_clf.fit(X_train, y_train)
        y_pred = lr_clf.predict(X_test)
        y_prob = lr_clf.predict_proba(X_test)[:, 0]
        y_prob = [(i, val, y_pred[i]) for i, val in enumerate(y_prob)]
        # pos and neg
        y_prob_neg = [val for val in y_prob if val[1] < 0.5]
        y_prob_neg = sorted(y_prob_neg, key=lambda x: x[1])
        # y_prob_P = sorted(y_prob_P, key=lambda x: x[1], reverse=True)
        print(y_prob_neg)
        y_prob_neg = y_prob_neg[:top_n]  # YZ: here is 0.00003, XXX
        y_prob_pos = [val for val in y_prob if val[1] >= 0.5]
        y_prob_pos = sorted(y_prob_pos, key=lambda x: x[1], reverse=True)
        # y_prob_N = sorted(y_prob_N, key=lambda x: x[1])
        print(y_prob_pos)  # YZ: here is 0.99, XXX
        y_prob_pos = y_prob_pos[:top_n]
        keep_index = [val[0] for val in y_prob_neg] + [val[0] for val in y_prob_pos]
        not_keep_index = [i for i in range(len(y_pred)) if i not in keep_index]
        if len(keep_index) + len(not_keep_index) != len(y_pred):
            raise ValueError('top_n error!')

        X_test_keep = [X_test[i] for i in keep_index]
        y_pred_keep = [y_pred[i] for i in keep_index]
        X_train = np.concatenate((X_train, X_test_keep), axis=0)
        y_train = np.concatenate((y_train, y_pred_keep), axis=0)
        y_pred_store += y_pred_keep
        y_test_store += [y_test[i] for i in keep_index]

        # yz added try except
        try:
            print('total:', len(y_pred_keep), 'accuracy',
                  round(accuracy_score(y_pred_keep, [y_test[i] for i in keep_index]), 2),
                  'true_true', sum([y_test[i] for i in keep_index]),
                  'min_P', round(max([val[1] for val in y_prob_neg]), 2),
                  'min_N', round(min([val[1] for val in y_prob_pos]), 2),
                  )
        except:
            pass
        X_test = [X_test[i] for i in not_keep_index]
        y_test = [y_test[i] for i in not_keep_index]
        if X_test == previous_r_target:
            break
        previous_r_target = X_test[:]
    if len(y_pred_store) != len(test_labels):
        raise ValueError('output dimension error!')
    output_score = [y_pred_store[i] == y_test_store[i] for i in range(len(y_test_store))]
    gradual_score = sum(output_score) / len(output_score)
    # original_score = S2T(train_features, train_labels, test_features, test_labels)
    lr_clf = model
    lr_clf.fit(X_train, y_train)
    y_pred = lr_clf.predict(test_features)
    lm_score = [y_pred[i] == test_labels[i] for i in range(len(test_labels))]
    lm_score = sum(lm_score) / len(lm_score)
    print(lm_score, gradual_score)

    return original_score, lm_score, gradual_score
