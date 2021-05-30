import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
import pickle
from bert_embedding import *
from dataset_multi import load_ner_data
from sklearn.model_selection import train_test_split
from src.domain_space_alignment import ht_lr
import sys

# embedding need 1) sentence id, 2) position, 3) bert word embedding


# TODO: Potentially rewrite this to a class
def multiclass_one_all(base_model, train_features, train_labels, test_features, test_labels, conf, ht=False):
    """

    :param base_model: logistic regression
    :param train_features: on the word level; each word has a set of features
    :param train_labels: word labels
    :param test_features:
    :param test_labels:
    :param ht: True or False
    :param conf: confidence level
    :return:
    """
    unique_labels = sorted(list(set(train_labels)))
    train_binary_labels = np.zeros(len(train_labels))
    test_binary_labels = np.zeros(len(test_labels))

    logs = []
    for label in unique_labels:
        train_binary_labels[train_labels == label] = 1
        base_model.fit(train_features, train_binary_labels)  # pseudo labels
        if ht:
            test_features = ht_lr(train_features, train_binary_labels, test_features, test_binary_labels)
        y_prob = base_model.predict_prob(test_features)  # probabilities
        y_combo = [(i, x) for i, x in enumerate(y_prob)]

        logs.append(y_prob)

    logs = np.array(logs)
    pred_logs = logs/np.sum(logs, 0)
    pred = np.argmax(pred_logs, 0)

    return pred
