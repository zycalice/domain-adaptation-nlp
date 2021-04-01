from transformers import DistilBertTokenizer, DistilBertModel
from utils import *


def output_bert_embeddings(domain_types, data_size):
    for domain in domain_types:
        _ = tokenize_encode_bert_sentences(tokenizer_d, model_d, list(eval("X_train_" + domain)[:data_size]),
                                           data_path + "all_bert/" + "encoded_" + domain + "_train_" + str(
                                               data_size))
        _ = tokenize_encode_bert_sentences(tokenizer_d, model_d, list(eval("X_dev_" + domain)[:data_size]),
                                           data_path + "all_bert/" + "encoded_" + domain + "_dev_" + str(data_size))


if __name__ == '__main__':
    data_path = "../data/"
    tokenizer_d = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_d = DistilBertModel.from_pretrained('distilbert-base-uncased')
    domains = ["tw", "az", "mv", "fi"]
    data_size = 3000

    for domain in domains:
        for data_type in ["train", "dev"]:
            file = "X_" + data_type + "_" + domain
            filename = data_path + "all_cleaned/" + file + ".npy"
            exec(file + " = np.load('" + filename + "', allow_pickle = True)")

    output_bert_embeddings(domains, data_size)
