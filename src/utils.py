import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# Helper functions.

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def load_np_files(data_path, domains, data_types, load_feature):
    np_dataset = {}
    for domain in domains:
        for data_type in data_types:
            if load_feature:
                file_prefix = "X_"
            else:
                file_prefix = "y_"
            file = file_prefix + data_type + "_" + domain
            filename = data_path + "all_cleaned/" + file + ".npy"
            np_dataset[file] = np.load(filename, allow_pickle=True)
    return np_dataset


def load_bert(data_path, domains, data_size):
    bert_dataset = {}
    for domain in domains:
        bert_dataset[domain] = np.load(data_path + "all_bert/encoded_" +
                                                        domain + "_train_" + str(data_size) + ".npy")
    return bert_dataset


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

def psuedo_labeling(X_source, y_source, X_ti, y_ti, model, conf=0):
    base_model = model
    model.fit(X_source, y_source)
    y_prob = base_model.predict_proba(X_ti)[:, 0]
    X_ti_keep = X_ti[(y_prob >= 0.5 + conf) | (y_prob < 0.5 - conf)]
    y_pred = model.predict(X_ti_keep)
    X_source_updated = np.concatenate((X_source, X_ti_keep), 0)
    y_source_updated = np.concatenate((y_source, y_pred), 0)
    return X_source_updated, y_source_updated


def gradual_train(X_source, y_source, X_target, y_target, base_model, dists, group_size=5, conf=0):
    # initiate model
    model = base_model

    # create groups within targets and gradually train
    dists = np.array(dists)
    dists_order = np.argsort(dists)
    dists_rank = np.argsort(dists_order)

    step = len(y_target) / group_size
    X_target_groups = []
    y_target_groups = []
    gradual_scores = []
    X_source_updated = X_source.copy()
    y_source_updated = y_source.copy()
    for i in range(group_size):
        subset_tf = (step * i <= dists_rank) & (dists_rank < step * (i + 1))
        X_ti = X_target[subset_tf]
        y_ti = y_target[subset_tf]
        X_target_groups.append(X_ti)
        y_target_groups.append(y_ti)
        X_source_updated, y_source_updated = psuedo_labeling(X_source_updated, y_source_updated, X_ti, y_ti, model,
                                                             conf)
        gradual_score = model.fit(X_source_updated, y_source_updated).score(X_target, y_target)
        gradual_scores.append(gradual_score)
    return gradual_scores


def gradual_train_groups(X_source_raw, y_source_raw, X_target_raw, y_target_raw, base_model, data_size, group_range,
                         plot_hist=True):
    # initial data and model
    print(data_size)
    X_source, y_source = X_source_raw[:data_size], y_source_raw[:data_size]
    X_target, y_target = X_target_raw[:data_size], y_target_raw[:data_size]
    print(X_source.shape, y_source.shape, X_target.shape, y_target.shape)

    model = base_model
    model.fit(X_source, y_source)
    no_self_train_adaptation_score = model.score(X_target, y_target)

    # calculate distances
    source_center = np.mean(X_source, 0)
    dists = [1 - cosine_similarity(source_center.reshape(1, -1), x.reshape(1, -1))[0][0] for x in X_target]
    if plot_hist:
        plt.hist(dists, bins=100)
        plt.show()

    # save accuracies
    final_accuracies = [no_self_train_adaptation_score]
    accuracies_ti = {"no_self_train_adaptation_score": no_self_train_adaptation_score}
    for i in range(group_range[0] + 1, group_range[1] + 1):
        # print("\ngroup = ", i)
        data_size = data_size
        base_model = base_model
        gradual_scores = gradual_train(
            X_source, y_source,
            X_target, y_target,
            base_model, dists=dists, group_size=i)
        final_accuracies.append(gradual_scores[-1])
        accuracies_ti[i] = gradual_scores
        print("group", i, '{:.2f}'.format(no_self_train_adaptation_score*100),
              ['{:.2f}'.format(elem*100) for elem in gradual_scores])

    return final_accuracies, accuracies_ti, dists
