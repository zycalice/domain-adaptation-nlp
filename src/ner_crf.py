from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support, make_scorer
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
import re
import pickle
from bert_embedding import *
from dataset import load_ner_data
from sklearn.model_selection import train_test_split

# # Load tokenizer and model
# tokenizer_d = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model_d = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load pre-computed bert embeddings.
data_path = "../data/"
with open(data_path + "wiki_sec_word2idx.json") as f:
    word2idx = json.load(f)

ner_bert = np.load("../data/all_bert/encoded_ner_corpus.npy")

assert(len(ner_bert) == len(word2idx))


def word2features(sent, i):
    """
    The function generates all features
    for the word at position i in the
    sentence.
    """
    word = sent[i][0]
    embedding = ner_bert[word2idx[word]]
    # embedding_orig = tokenize_encode_bert_sentences_sample(tokenizer_d, model_d, word)[0]
    # if (embedding != embedding_orig).any():
    #     print(word, word2idx[word]-1)
    #     raise ValueError
    features = {}
    for j in range(len(embedding)):
        features[str(j)] = embedding[j]
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [t[-1] for t in sent]


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


def run_crf(train_data, dev_data, model, output_name, crf_f1_report=True, crf_transition_analysis=False,
            output_predictions=False):
    # Load the training data
    train_sents = train_data
    dev_sents = dev_data

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_dev = [sent2features(s) for s in dev_sents]
    y_dev = [sent2labels(s) for s in dev_sents]

    crf = model
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')  # why remove 0?
    y_pred_train = crf.predict(X_train)
    y_pred_dev = crf.predict(X_dev)

    crf.fit(X_train + X_dev, y_train + y_dev)

    # f1 score different way
    if crf_f1_report:
        # metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
        # group B and I results
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        print(metrics.flat_classification_report(
            y_train, y_pred_train, labels=sorted_labels, digits=3
        ))

        print(metrics.flat_classification_report(
            y_dev, y_pred_dev, labels=sorted_labels, digits=3
        ))

    if crf_transition_analysis:
        print("Top likely transitions:")
        print_transitions(Counter(crf.transition_features_).most_common(20))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    print("Saving model")
    model_filename = '../outputs/ner_model.sav'
    pickle.dump(model, open(model_filename, 'wb'))

    if output_predictions:
        print("Writing to results.txt")
        output_predictions_to_file(dev_sents, output_name, y_pred_dev)

    # print("Now run: python conlleval.py results.txt")


if __name__ == '__main__':
    wiki = load_ner_data("../data/ner_wikigold/wikigold.conll.txt", " ")[:-1]
    sec = load_ner_data("../data/ner_sec/FIN5.txt")[:-1]

    train_wiki, test_wiki = train_test_split(wiki, random_state=7)
    train_sec, test_sec = train_test_split(sec, random_state=7)

    crf = sklearn_crfsuite.CRF(
        c1=0.1,
        c2=0.2,
        algorithm='lbfgs',
        max_iterations=200,
        all_possible_transitions=True,
        # all_possible_states=True,
    )

    run_crf(train_sec, test_sec, crf, "../outputs/wiki_sec_crf_results.txt",
            crf_f1_report=True, crf_transition_analysis=False, output_predictions=False)
