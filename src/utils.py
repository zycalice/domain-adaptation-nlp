import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Helper functions.

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


# BERT embedding functions.

def tokenize_encode_bert_sentences(tokenizer, model, input_sentences, output_path):
    output = np.zeros([len(input_sentences), 768])
    for i, x in enumerate(input_sentences):
        output[i] = tokenize_encode_bert_sentences_sample(tokenizer, model, [x])
    np.save(output_path, output)
    return output


def tokenize_encode_bert_sentences_sample(tokenizer, model, input_sentences):
    encoded_input = tokenizer(input_sentences, return_tensors='pt', truncation=True, padding=True)
    output = model(**encoded_input)[0][:, 0, :].detach().numpy()
    return output


# Pseudo labeling (self train) and gradual train.

def psuedo_labeling(X_source, y_source, X_ti, y_ti, model):  # incorporate alpha in psuedo labeling later
    model = model
    model.fit(X_source, y_source)
    y_pred = model.predict(X_ti)
    X_source_updated = np.concatenate((X_source, X_ti), 0)
    y_source_updated = np.concatenate((y_source, y_pred), 0)
    return X_source_updated, y_source_updated


def gradual_train(X_source, y_source, X_target, y_target, base_model, data_size=2000, group_size=5):
    # initial model:
    model = base_model
    model.fit(X_source, y_source)
    original = model.score(X_target, y_target)

    # calculate distances
    source_center = np.mean(X_source, 0)
    dists = [1 - cosine_similarity(source_center.reshape(1, -1), x.reshape(1, -1))[0][0] for x in X_target]
    plt.hist(dists)
    plt.show()

    # create groups within targets and gradually train
    dists = np.array(dists)
    dists_order = np.argsort(dists)
    dists_rank = np.argsort(dists_order)

    step = data_size / group_size
    X_target_groups = []
    y_target_groups = []
    X_source_updated = X_source
    y_source_updated = y_source
    for i in range(group_size):
        subset_tf = (step * i <= dists_rank) & (dists_rank < step * (i + 1))
        X_ti = X_target[:data_size][subset_tf]
        y_ti = y_target[:data_size][subset_tf]
        X_target_groups.append(X_ti)
        y_target_groups.append(y_ti)
        X_source_updated, y_source_updated = psuedo_labeling(X_source_updated, y_source_updated, X_ti, y_ti, model)
        print(model.fit(X_source_updated, y_source_updated).score(X_target, y_target))

    model.fit(X_source_updated, y_source_updated)
    gradual = model.score(X_target, y_target)

    return original, gradual

