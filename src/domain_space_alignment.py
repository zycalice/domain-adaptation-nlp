# coding: utf-8
import math
import random
from itertools import permutations
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

max_iteration=2000000
# all train_features, test_features needs to be in numpy format
def ht_lr(train_features, train_labels, test_features, top_threshold):
    # aligning target domain to source domain
    lr_clf = LogisticRegression(max_iter=max_iteration)
    lr_clf.fit(train_features, train_labels)
    y_pred = lr_clf.predict(test_features)
    y_prob = lr_clf.predict_proba(test_features)[:, 1]
    y_prob = [(i, val, y_pred[i]) for i, val in enumerate(y_prob)]
    y_prob = sorted(y_prob, key=lambda x: x[1])
    # y_prob_P = y_prob[:int(len(test_labels) * fraction)]  # TODO not enough enough, the threshold is too large
    # y_prob_N = y_prob[-int(len(test_labels) * fraction):]
    y_prob_P = y_prob[-top_threshold:]
    y_prob_N = y_prob[:top_threshold]

    sourcePos = [val for i, val in enumerate(train_features) if train_labels[i] == 1]
    sourceNeg = [val for i, val in enumerate(train_features) if train_labels[i] == 0]
    targetPos = [test_features[val[0]] for val in y_prob_P]
    targetNeg = [test_features[val[0]] for val in y_prob_N]
    v = np.mean(sourcePos, axis=0) - np.mean(sourceNeg, axis=0)
    u = np.mean(targetPos, axis=0) - np.mean(targetNeg, axis=0)
    c1 = np.mean(test_features, axis=0)
    c2 = np.mean(np.concatenate([sourcePos, sourceNeg], axis=0), axis=0)

    test_features = hh_lr(u, v, c1, c2, test_features)
    return test_features


def hh_lr(u, v, c1, c2, points):
    # household transformation
    u_mag = np.linalg.norm(u)
    u_unit = u / u_mag

    v_mag = np.linalg.norm(v)
    v_unit = v / v_mag

    # Scaling so pos-neg vectors have the same magnitude
    scaled_points = points * v_mag / u_mag
    scaled_c1 = c1 * v_mag / u_mag

    # gettinng dimension of vector space
    k = len(c2)

    # calculating isometric linear transformation: householder transformation
    A = np.eye(k) - (2 * (np.outer(u_unit - v_unit, u_unit - v_unit) / np.inner(u_unit - v_unit, u_unit - v_unit)))

    # applying isometric transformation
    points_after_isometric = scaled_points @ A.T
    c1_after_isometric = scaled_c1 @ A.T

    # translation
    points_after_translation = points_after_isometric + (c2 - c1_after_isometric)

    return points_after_translation


def S2T_p_hh(train_features, train_labels, test_features, test_labels, top_threshold):
    # Domain Space Alignment model
    lowerbound = S2T(train_features, train_labels, test_features, test_labels)
    test_features = ht_lr(train_features, train_labels, test_features, top_threshold)

    lr_clf = LogisticRegression(max_iter=max_iteration)
    lr_clf.fit(train_features, train_labels)

    return lowerbound, lr_clf.score(test_features, test_labels)


def S2T(train_features, train_labels, test_features, test_labels):
    # lowerbound model
    lr_clf = LogisticRegression(max_iter=max_iteration)
    lr_clf.fit(train_features, train_labels)
    return lr_clf.score(test_features, test_labels)


def cv_nfold_blc(func, all_data, nfold=10, top_threshold=100):
    # n fold validation
    # labels are balanced

    data_permu = list(permutations(all_data, 2))
    S2T_scores = []
    func_scores = []
    # dist_list = []

    for index, permu in enumerate(data_permu):
        source_data = permu[0]
        target_data = permu[1]
        y_source = source_data[1][:]
        y_target = target_data[1][:]
        X_i_source_p = [(i, val) for i, val in enumerate(source_data[0]) if y_source[i] == 1]
        X_i_source_n = [(i, val) for i, val in enumerate(source_data[0]) if y_source[i] == 0]
        X_i_target = [(i, val) for i, val in enumerate(target_data[0])]
        random.Random(0).shuffle(X_i_source_p)
        random.Random(0).shuffle(X_i_source_n)
        X_i_source = []
        for num in range(len(X_i_source_p) + len(X_i_source_n)):
            if (num % 2) == 0:
                X_i_source.append(X_i_source_p[int(num / 2)])
            else:
                X_i_source.append(X_i_source_n[int((num - 1) / 2)])
        random.Random(0).shuffle(X_i_target)
        X_source = [[val[1]
                     for val in X_i_source[
                                math.ceil(len(X_i_source) / nfold) * thresh:math.ceil(len(X_i_source) / nfold) * (
                                            1 + thresh)]]
                    for thresh in range(nfold)
                    ]
        y_source = [[y_source[val[0]]
                     for val in X_i_source[
                                math.ceil(len(X_i_source) / nfold) * thresh:math.ceil(len(X_i_source) / nfold) * (
                                            1 + thresh)]]
                    for thresh in range(nfold)
                    ]
        X_target = [[val[1]
                     for val in X_i_target[
                                math.ceil(len(X_i_target) / nfold) * thresh:math.ceil(len(X_i_target) / nfold) * (
                                            1 + thresh)]]
                    for thresh in range(nfold)
                    ]
        y_target = [[y_target[val[0]]
                     for val in X_i_target[
                                math.ceil(len(X_i_target) / nfold) * thresh:math.ceil(len(X_i_target) / nfold) * (
                                            1 + thresh)]]
                    for thresh in range(nfold)
                    ]
        if nfold==1:
            raise ValueError
        else:
            for fold in range(nfold):
                input_X_source = X_source[fold]
                input_y_source = y_source[fold]
                input_X_target = np.concatenate([val for i, val in enumerate(X_target) if i != fold], axis=0)
                input_y_target = np.concatenate([val for i, val in enumerate(y_target) if i != fold], axis=0)

                S2T_scr, func_scr = func(input_X_source, input_y_source, input_X_target, input_y_target, top_threshold)
                S2T_scores.append(S2T_scr)
                func_scores.append(func_scr)
        print(index)

    accuracy_gain = [func_scores[i] - S2T_scores[i] for i in range(len(func_scores))]
    return accuracy_gain, S2T_scores, func_scores


if __name__ == '__main__':
    pass

    amazon_data_path = "../data/amazon_reviews/amazon_4.pickle"
    with open(amazon_data_path, "rb") as fr:
        all_data = pickle.load(fr)

    accuracy_gain, S2T_scores, func_scores = cv_nfold_blc(S2T_p_hh, all_data, nfold=5, top_threshold=200)

    print('lowerbound score:', np.mean(S2T_scores))
    print('Domain Space Alignment model score:', np.mean(func_scores))
