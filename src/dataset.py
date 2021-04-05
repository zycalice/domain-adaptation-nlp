import os
import pandas as pd
import zipfile
from utils import *
import bz2
from sklearn.model_selection import train_test_split


def get_label_feature_amazon(file):
    labels = []
    reviews = []
    for i, e in enumerate(file):
        sep_pos = e.find(" ")
        label = 1 if e[:sep_pos] == "__label__2" else 0
        review = e[(sep_pos + 1):]
        labels.append(label)
        reviews.append(review)
    return np.array(reviews), np.array(labels)


if __name__ == '__main__':
    data_path = "../data/"

    # Twitter sentiment.
    twitter_zip = zipfile.ZipFile(data_path + "twitter/twitter.zip")
    twitter = pd.read_csv(twitter_zip.open('training.1600000.processed.noemoticon.csv'), encoding='ISO-8859-1',
                          header=None)

    y_tw = twitter[0].values
    y_tw[y_tw == 4] = 1
    X_tw = twitter[5].values

    X_train_val_tw, X_test_tw, y_train_val_tw, y_test_tw = train_test_split(X_tw, y_tw, test_size=0.33, random_state=7)
    X_train_tw, X_dev_tw, y_train_tw, y_dev_tw = train_test_split(X_train_val_tw, y_train_val_tw, test_size=0.33,
                                                                  random_state=7)

    # Amazon sentiment.
    amazon_train_file = bz2.BZ2File(data_path + 'amazon/train.ft.txt.bz2').readlines()
    amazon_test_file = bz2.BZ2File(data_path + 'amazon/test.ft.txt.bz2').readlines()
    amazon_train_file = [x.decode('utf-8') for x in amazon_train_file]
    amazon_test_file = [x.decode('utf-8') for x in amazon_test_file]

    X_train_val_az, y_train_val_az = get_label_feature_amazon(amazon_train_file)
    X_test_az, y_test_az = get_label_feature_amazon(amazon_train_file)
    X_train_az, X_dev_az, y_train_az, y_dev_az = train_test_split(X_train_val_az, y_train_val_az, test_size=0.33,
                                                                  random_state=7)

    # Movie sentiment.
    movie_zip = zipfile.ZipFile(data_path + "movies/movie.zip")
    movie_train = pd.read_csv(movie_zip.open('Train.csv'))
    movie_dev = pd.read_csv(movie_zip.open('Valid.csv'))
    movie_test = pd.read_csv(movie_zip.open('Test.csv'))

    X_train_mv, X_dev_mv, X_test_mv = movie_train['text'].values, movie_dev['text'].values, movie_test['text'].values
    y_train_mv, y_dev_mv, y_test_mv = movie_train['label'].values, movie_dev['label'].values, movie_test['label'].values

    # Finance sentiment.
    finance_file = pd.read_csv(data_path + "/finance/archive/all-data.csv", encoding='ISO-8859-1', header=None)
    finance = finance_file[finance_file[0] != "neutral"]

    y_fi = finance[0].values
    y_fi[y_fi == "negative"] = 0
    y_fi[y_fi == "positive"] = 1
    y_fi = y_fi.astype("int64")
    X_fi = finance[1].values

    X_train_val_fi, X_test_fi, y_train_val_fi, y_test_fi = train_test_split(X_fi, y_fi, test_size=0.33, random_state=7)
    X_train_fi, X_dev_fi, y_train_fi, y_dev_fi = train_test_split(X_train_val_fi, y_train_val_fi, test_size=0.1,
                                                                  random_state=7)

    # output data for further use
    for data in [X_train_tw, y_train_tw, X_dev_tw, y_dev_tw, X_test_tw, y_test_tw,
                 X_train_az, y_train_az, X_dev_az, y_dev_az, X_test_az, y_test_az,
                 X_train_mv, y_train_mv, X_dev_mv, y_dev_mv, X_test_mv, y_test_mv,
                 X_train_fi, y_train_fi, X_dev_fi, y_dev_fi, X_test_fi, y_test_fi]:
        file_path = data_path + namestr(data, globals())
        if not os.path.isfile(file_path):
            np.save(data_path + "all_cleaned/" + namestr(data, globals()), data)

    # for data in [X_train_fi, y_train_fi, X_dev_fi, y_dev_fi, X_test_fi, y_test_fi]:
    #     np.save(data_path + "all_cleaned/" + namestr(data, globals()), data)
