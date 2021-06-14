import numpy as np
import json
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from src.utils import *
from src.bert_embedding import *
from transformers import BertTokenizer, BertModel


if __name__ == '__main__':
    data_main_path = "../data/movie_reviews/aclImdb/"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

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
    bert_train = tokenize_encode_bert_sentences_batch(tokenizer, model, list(X_train_val[:2000]),
                                                "../outputs/" + "encoded_aclimbd_train_2000")
    bert_test = tokenize_encode_bert_sentences_batch(tokenizer, model, list(X_test[:2000]),
                                               "../outputs/" + "encoded_aclimbd_test_2000")

    aclimbd_array = np.array([bert_train, y_train_val[:2000], "aclimbd"])
    np.save(aclimbd_array, data_main_path + "movie_review.npy")
