import numpy as np
import json
import re
from datasets import load_dataset


# load data
def load_ner_data(path, separator=" "):
    with open(path) as f:
        text = f.read().split("\n\n")

    output = []
    for line in text:
        feature_label = []
        line = line.split("\n")
        for entry in line:
            feature_label.append(tuple(entry.split(separator)))
        output.append(feature_label)
    return output


# get words and tags
def unique_words_tags(data):
    unique_words = []
    unique_tags = []
    for sent in data:
        unique_words.extend(list(set(np.array(sent)[:, 0])))
        unique_tags.extend(list(set(np.array(sent)[:, -1])))

    return set(unique_words), set(unique_tags)


# get words and tags distributions
def distributions_words_tags(data_input):
    unique_words = {}
    unique_tags = {}
    for i in range(len(data_input) - 1):
        sent = data_input[i]
        for t in sent:
            word = t[0]
            tag = t[-1]

            if word in unique_words:
                unique_words[word] += 1
            else:
                unique_words[word] = 1

            if tag in unique_tags:
                unique_tags[tag] += 1
            else:
                unique_tags[tag] = 1

    return sorted(unique_words), sorted(unique_tags)


def sent_to_tuple(sent, label_list, pos_list):
    ner_tags = sent['ner_tags']
    pos_tags = sent['pos_tags']
    tokens = sent['tokens']
    sent_list = []
    for i in range(len(sent['ner_tags'])):
        sent_list.append((tokens[i], pos_list[pos_tags[i]], label_list[ner_tags[i]]))
    return sent_list


# transform data
def transform_label(sent):
    return [(t[0], re.sub("E-", "I-", re.sub("S-", "B-", t[1]))) for t in sent]


if __name__ == '__main__':
    data_path = "../data/"

    # NER v1.
    wiki = load_ner_data("../data/ner_wikigold/wikigold.conll.txt", " ")[:-1]
    sec = load_ner_data("../data/ner_sec/FIN5.txt")[:-1]

    words_wiki, tags = unique_words_tags(wiki)
    words_sec, _ = unique_words_tags(sec)
    words = list(words_wiki | words_sec)
    words.sort()

    word2idx = {w: i for i, w in enumerate(words)}
    with open(data_path + "wiki_sec_word2idx.json", "w") as outfile:
        json.dump(word2idx, outfile, indent=4)

    # NER v2.
    dataset = load_dataset('conll2003')
    label_list_conll = dataset['train'].features['ner_tags'].feature.names
    pos_list_conll = dataset['train'].features['pos_tags'].feature.names
    conll2003 = [sent_to_tuple(dataset['train'][x], label_list_conll, pos_list_conll)
                 for x in range(len(dataset['train']))]

    tech = load_ner_data(
        "/Users/yuchen.zhang/Documents/Projects/domain-adaptation-nlp/data/ner_tech/tech_test.txt"
    )
    tech = [transform_label(x) for x in tech]

    words_conll, conll_tags = unique_words_tags(conll2003)
    print(conll_tags)
    words_tech, tech_tags = unique_words_tags(tech)
    print(tech_tags)
    words_conll_format = list(words_conll | words_tech)
    words_conll_format.sort()

    conll_tech_word2idx = {w: i for i, w in enumerate(words_conll_format)}
    with open(data_path + "conll_tech_word2idx.json", "w") as outfile:
        json.dump(conll_tech_word2idx, outfile, indent=4)
