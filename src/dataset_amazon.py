import json
from os import listdir
from os.path import isfile, join


def load_data(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    reviews = text.split("\n")
    reviews = [i for i in reviews if (i != "") and ("UNCONFIDENT_INTENT_FROM_SLAD" not in i)]
    x = [x.split("\t")[0] for x in reviews]
    y = [x.split("\t")[1] for x in reviews]
    return x, y


if __name__ == '__main__':

    freq_used = ['Home_and_Kitchen', 'Books', 'Electronics', 'Movies_and_TV']

    # read and clean
    data_path = "../data/amazon_reviews/amazon_review/"

    all_data = {}
    for folder_name in ["train", "dev", "test"]:
        folder_path = data_path + folder_name + "/"
        only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        for file in only_files:
            if (".test" in file) or (".train" in file) or (".test" in file):
                domain = file.split(".")[0]
                if domain in freq_used:
                    all_data[file] = load_data(folder_path + "/" + file)

    # output clean data
    with open("../data/all_cleaned/amazon_data_dict.txt", "w") as f:
        json.dump(all_data, f, indent=4)
