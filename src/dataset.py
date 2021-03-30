import numpy as np
import pandas as pd
import torch
import zipfile
import utils
import bz2
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Twitter sentiment.
    data_path = "../data/"
    twitter_zip = zipfile.ZipFile(data_path + "twitter.zip")
    twitter = pd.read_csv(twitter_zip.open('training.1600000.processed.noemoticon.csv'), encoding='ISO-8859-1',
                          header=None)

    y_tw = twitter[0].values
    y_tw[y_tw == 4] = 1
    X_tw = twitter[5].values

    X_train_val_tw, X_test_tw, y_train_val_tw, y_test_tw = train_test_split(X_tw, y_tw, test_size=0.33, random_state=7)
    X_train_tw, X_dev_tw, y_train_tw, y_dev_tw = train_test_split(X_train_val_tw, y_train_val_tw, test_size=0.33,
                                                                  random_state=7)

    for data in [X_train_tw, y_train_tw, X_dev_tw, y_dev_tw, X_test_tw, y_test_tw]:
        np.save(data_path + utils.namestr(data, globals()) + ".txt", data)
