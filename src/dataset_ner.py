import numpy


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

    return unique_words, unique_tags