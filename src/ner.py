# import numpy as np
import sklearn_crfsuite
from sklearn.linear_model import LogisticRegression
from sklearn_crfsuite import metrics
from collections import Counter
import pickle
from src.dataset_ner import load_ner_data
from sklearn.model_selection import train_test_split
import sys
from src.dataset_ner import *
from src.bert_embedding import *

# Load pre-computed bert embeddings. #TODO sub-token
data_path = "../data/"
with open(data_path + "conll_tech_word2idx.json") as f:
    word2idx = json.load(f)

# with open(data_path + "wiki_sec_word2idx.json") as f:
#     word2idx = json.load(f)

# ner_bert = np.load("../data/all_bert/bert_cased_encoded_ner_corpus_conll.npy")
# ner_bert = np.load("../data/all_bert/bert_ner_encoded_ner_corpus_conll.npy")
ner_bert = np.load("../data/all_bert/bert_ner_first_token_encoded_ner_corpus_conll.npy")

# cased result is better than uncased for crf

assert (len(ner_bert) == len(word2idx))


def word2features(sent, i, use_crf):
    """
    The function generates all features
    for the word at position i in the
    sentence.
    """
    word = sent[i][0]
    if word == "":
        print("the sent is", sent)
    embedding = ner_bert[word2idx[word]]
    # embedding_orig = tokenize_encode_bert_sentences_sample(tokenizer_d, model_d, word)[0]
    # if (embedding != embedding_orig).any():
    #     print(word, word2idx[word]-1)
    #     raise ValueError
    features = {}

    # if not use_crf:
    #     # embedding need 1) position or relative position, 2) (fine-tuned or not) bert word embedding,
    #     # TODO add sentence id; may or may not need position; +-1 context; first word, or last word
    #     embedding = np.concatenate((embedding, [i]), 0)

    for j in range(len(embedding)):
        features[str(j)] = embedding[j]

    return features


def sent2features(sent, use_crf):
    return [word2features(sent, i, use_crf) for i in range(len(sent))]


def sent2labels(sent):
    return [t[-1] for t in sent]


def get_features_labels(train_sents, test_sents, use_crf):
    x_train = [sent2features(s, use_crf) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    x_test = [sent2features(s, use_crf) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    return x_train, y_train, x_test, y_test


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def output_predictions_to_file(sents, output_name, y_pred):
    # format is: word gold pred

    with open("../outputs/predictions/" + output_name, "w") as out:
        for j, sent in enumerate(sents):
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j][i]
                out.write("{}\t{}\t{}\n".format(word, gold, pred))

        out.write("\n")


def run_crf(train_data, dev_data, base_model, output_name=None, crf_f1_report=True,
            crf_transition_analysis=False, output_predictions=False, save_model=False):
    # Load the training data
    train_sents = train_data
    dev_sents = dev_data

    x_train, y_train, x_dev, y_dev = get_features_labels(train_sents, dev_sents, True)

    crf = base_model
    base_model.fit(x_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')  # why remove 0?
    y_pred_train = crf.predict(x_train)
    y_pred_dev = crf.predict(x_dev)
    print(y_pred_dev[:10])

    # f1 score different way
    if crf_f1_report:
        # metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
        # group B and I results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        print("== Train ==")
        print(metrics.flat_classification_report(
            y_train, y_pred_train, labels=sorted_labels, digits=3
        ))

        print("== Dev ==")
        print(metrics.flat_classification_report(
            y_dev, y_pred_dev, labels=sorted_labels, digits=3
        ))

    if crf_transition_analysis:
        print("Top likely transitions:")
        print_transitions(Counter(crf.transition_features_).most_common(20))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    if save_model:
        print("Saving model")
        model_filename = '../outputs/ner_model.sav'
        pickle.dump(crf, open(model_filename, 'wb'))

    if output_predictions:
        print("Writing to results.txt")
        output_predictions_to_file(dev_sents, output_name, y_pred_dev)


def run_multiclass(train_data, dev_data, base_model, conf, test_ht, output_name=None, f1_report=True,
                   output_predictions=False, save_model=False):
    # Load the training data
    train_sents = train_data
    dev_sents = dev_data

    x_train, y_train, x_dev, y_dev = get_features_labels(train_sents, dev_sents, False)

    # flat the data
    x_train_multiclass = [list(x.values()) for sent in x_train for x in sent]
    y_train_multiclass = [y for sent in y_train for y in sent]
    x_dev_multiclass = [list(x.values()) for sent in x_dev for x in sent]
    y_dev_multiclass = [y for sent in y_dev for y in sent]
    print(np.array(x_train_multiclass).shape)

    train_idx = [i for i, sent in enumerate(y_train) for _ in sent]
    dev_idx = [i for i, sent in enumerate(y_dev) for _ in sent]

    labels = sorted(list(set([x for x in y_dev_multiclass])))
    labels.remove('O')  # why remove 0?

    # predictions
    y_pred_train_list = multiclass_self_train(
        base_model,
        x_train_multiclass, y_train_multiclass,
        x_train_multiclass, y_train_multiclass,
        conf, False)

    y_pred_dev_list = multiclass_self_train(
        base_model,
        x_train_multiclass, y_train_multiclass,
        x_dev_multiclass, y_dev_multiclass,
        conf, test_ht)

    # print("train", np.array(y_pred_train_list).shape, "dev", np.array(y_pred_dev_list).shape)

    y_pred_train = []
    y_pred_dev = []

    # convert from list to list of lists
    # train
    sent = []
    for i, sent_idx in enumerate(train_idx):
        word = y_pred_train_list[i]
        if (i > 0) and (sent_idx != train_idx[i - 1]):
            y_pred_train.append(sent)
            sent = [word]
        else:
            sent.append(word)
    y_pred_train.append(sent)

    # test
    sent = []
    for i, sent_idx in enumerate(dev_idx):
        word = y_pred_dev_list[i]
        if (i > 0) and (sent_idx != dev_idx[i - 1]):
            y_pred_dev.append(sent)
            sent = [word]
        else:
            sent.append(word)
    y_pred_dev.append(sent)

    # f1 score different way
    if f1_report:
        # metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
        # group B and I results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        print("== Train ==")
        print(metrics.flat_classification_report(
            y_train, y_pred_train, labels=sorted_labels, digits=3
        ))

        print("== Dev ==")
        print(metrics.flat_classification_report(
            y_dev, y_pred_dev, labels=sorted_labels, digits=3
        ))

    if output_predictions:
        print("Writing to results.txt")
        output_predictions_to_file(dev_sents, output_name, y_pred_dev)


def run_all_multiclass_experiments(output_file_name="../outputs/" + "ner_cased" + '.txt'):
    # In domain crf
    sys.stdout = open(output_file_name, 'w')
    print("\nIn domain: train_sec, test_sec")
    run_crf(train_sec, test_sec, crf_model,
            crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)

    print("\nIn domain: train_wiki, test_wiki")
    run_crf(train_wiki, test_wiki, crf_model,
            crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)

    # Out domain crf
    print("\nOut domain: train_wiki, test_sec")
    run_crf(train_wiki, test_sec, crf_model,
            crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)

    print("\nOut domain: train_sec, test_wiki")
    run_crf(train_sec, test_wiki, crf_model,
            crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)

    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # In domain multiclass
    sys.stdout = open("../outputs/" + "ner_cased_multiclass" + '.txt', 'w')
    print("\nIn domain multiclass: train_sec, test_sec")
    run_multiclass(train_sec, test_sec, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    print("\nIn domain multiclass: train_wiki, test_wiki")
    run_multiclass(train_wiki, test_wiki, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    # Out domain multiclass
    print("\nOut domain multiclass: train_wiki, test_sec")
    run_multiclass(train_wiki, test_sec, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    print("\nOut domain multiclass: train_sec, test_wiki")
    run_multiclass(train_sec, test_wiki, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    # Out domain multiclass HT
    print("\nOut domain multiclass HT: train_wiki, test_sec")
    run_multiclass(train_wiki, test_sec, lr_model, test_ht=True, conf=None,
                   f1_report=True, output_predictions=False)

    print("\nOut domain multiclass HT: train_sec, test_wiki")
    run_multiclass(train_sec, test_wiki, lr_model, test_ht=True, conf=None,
                   f1_report=True, output_predictions=False)

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    wiki = load_ner_data("../data/ner_wikigold/wikigold.conll.txt", " ")[:-1]
    sec = load_ner_data("../data/ner_sec/FIN5.txt")[:-1]

    dataset = load_dataset('conll2003')
    label_list_input = dataset['train'].features['ner_tags'].feature.names
    pos_list_input = dataset['train'].features['pos_tags'].feature.names
    conll2003 = [sent_to_tuple(dataset['train'][x], label_list_input, pos_list_input) for x in range(len(dataset['train']))]

    tech = load_ner_data(
        "/Users/yuchen.zhang/Documents/Projects/domain-adaptation-nlp/data/ner_tech/tech_test.txt"
    )
    tech = [transform_label(x) for x in tech]

    train_wiki, test_wiki = train_test_split(wiki, random_state=7)
    train_sec, test_sec = train_test_split(sec, random_state=7)
    train_conll, test_conll = train_test_split(conll2003, random_state=7)
    train_tech, test_tech = train_test_split(tech, random_state=7)

    # model
    crf_model = sklearn_crfsuite.CRF(
        c1=0.1,
        c2=0.2,
        algorithm='lbfgs',
        max_iterations=200,
        all_possible_transitions=True,
        # all_possible_states=True,
    )

    lr_model = LogisticRegression(max_iter=20000000)

    # # In domain crf
    # sys.stdout = open("../outputs/" + "ner_cased" + '.txt', 'w')
    # print("\nIn domain: train_sec, test_sec")
    # run_crf(train_sec, test_sec, crf_model,
    #         crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)
    #
    # print("\nIn domain: train_wiki, test_wiki")
    # run_crf(train_wiki, test_wiki, crf_model,
    #         crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)
    #
    # # Out domain crf
    # print("\nOut domain: train_wiki, test_sec")
    # run_crf(train_wiki, test_sec, crf_model,
    #         crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)
    #
    # print("\nOut domain: train_sec, test_wiki")
    # run_crf(train_sec, test_wiki, crf_model,
    #         crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)
    #
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__

    # In domain multiclass using wiki and sec.

    # sys.stdout = open("../outputs/" + "ner_cased_multiclass_wiki_sec" + '.txt', 'w')
    # print("\nIn domain multiclass: train_sec, test_sec")
    # run_multiclass(train_sec, test_sec, lr_model, test_ht=False, conf=None,
    #                f1_report=True, output_predictions=False)
    #
    # print("\nIn domain multiclass: train_wiki, test_wiki")
    # run_multiclass(train_wiki, test_wiki, lr_model, test_ht=False, conf=None,
    #                f1_report=True, output_predictions=False)
    #
    # # Out domain multiclass
    # print("\nOut domain multiclass: train_wiki, test_sec")
    # run_multiclass(train_wiki, test_sec, lr_model, test_ht=False, conf=None,
    #                f1_report=True, output_predictions=False)
    #
    # print("\nOut domain multiclass: train_sec, test_wiki")
    # run_multiclass(train_sec, test_wiki, lr_model, test_ht=False, conf=None,
    #                f1_report=True, output_predictions=False)
    #
    # # Out domain multiclass HT
    # print("\nOut domain multiclass HT: train_wiki, test_sec")
    # run_multiclass(train_wiki, test_sec, lr_model, test_ht=True, conf=None,
    #                f1_report=True, output_predictions=False)
    #
    # print("\nOut domain multiclass HT: train_sec, test_wiki")
    # run_multiclass(train_sec, test_wiki, lr_model, test_ht=True, conf=None,
    #                f1_report=True, output_predictions=False)
    #
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__

    # In domain multiclass using conll and tech.

    sys.stdout = open("../outputs/" + "ner_ner_first_token_multiclass_conll_tech" + '.txt', 'w')
    print("\nIn domain multiclass: train_conll, test_conll")
    run_multiclass(train_conll, test_conll, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    print("\nIn domain multiclass: train_tech, test_tech")
    run_multiclass(train_tech, test_tech, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    # Out domain multiclass
    print("\nOut domain multiclass: train_conll, test_tech")
    run_multiclass(train_conll, test_tech, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    print("\nOut domain multiclass: train_tech, test_conll")
    run_multiclass(train_tech, test_conll, lr_model, test_ht=False, conf=None,
                   f1_report=True, output_predictions=False)

    # Out domain multiclass HT
    print("\nOut domain multiclass HT: train_conll, test_tech")
    run_multiclass(train_conll, test_tech, lr_model, test_ht=True, conf=None,
                   f1_report=True, output_predictions=False)

    print("\nOut domain multiclass HT: train_tech, test_conll")
    run_multiclass(train_tech, test_conll, lr_model, test_ht=True, conf=None,
                   f1_report=True, output_predictions=False)

    sys.stdout.close()
    sys.stdout = sys.__stdout__

