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


def multiclass_one_all(base_model, features, labels):
    """

    :param base_model: logistic regression
    :param features: on the word level; each word has a set of features
    :param labels: word labels
    :return:
    """
    unique_labels = sorted(list(set(labels)))
    binary_labels = np.zeros(len(labels))
    for label in unique_labels:
        binary_labels[labels == label] = 1
        base_model.fit(features, binary_labels)
        base_model.predict_prob(features)
    pass
