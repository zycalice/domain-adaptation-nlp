import numpy as np
import json
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from src.utils import *


if __name__ == '__main__':
    data_main_path = "../data/movie_reviews/aclImdb/"

    # get file names
    train_pos_files = [f for f in listdir(data_main_path + "train/pos/") if
                       isfile(join(data_main_path + "train/pos/", f))]
    train_neg_files = [f for f in listdir(data_main_path + "train/neg/") if
                       isfile(join(data_main_path + "train/neg/", f))]
    test_pos_files = [f for f in listdir(data_main_path + "test/pos/") if
                      isfile(join(data_main_path + "test/pos/", f))]
    test_neg_files = [f for f in listdir(data_main_path + "test/neg/") if
                      isfile(join(data_main_path + "test/neg/", f))]

    # read files by type
    train_pos_all = []
    for i in train_pos_files:
        with open(data_main_path + "train/pos/" + i) as f:
            train_pos_all.append(f.read())

    train_neg_all = []
    for i in train_neg_files:
        with open(data_main_path + "train/neg/" + i) as f:
            train_neg_all.append(f.read())

    test_pos_all = []
    for i in test_pos_files:
        with open(data_main_path + "test/pos/" + i) as f:
            test_pos_all.append(f.read())

    test_neg_all = []
    for i in test_neg_files:
        with open(data_main_path + "test/neg/" + i) as f:
            test_neg_all.append(f.read())

    print(len(train_pos_all), len(train_neg_all), len(test_pos_all), len(test_neg_all))

    # concat data
    all_data = train_pos_all + train_neg_all + test_pos_all + test_neg_all
    all_label = [2 for _ in range(len(train_pos_all))] + \
                [1 for _ in range(len(train_neg_all))] + \
                [2 for _ in range(len(test_pos_all))] + \
                [1 for _ in range(len(test_neg_all))]

    # self define train test with shuffle
    X_train_val, X_test, y_train_val, y_test = train_test_split(all_data, all_label, test_size=0.33, random_state=7)
