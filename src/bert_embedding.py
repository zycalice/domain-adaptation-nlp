from transformers import DistilBertTokenizer, DistilBertModel
from utils import *


def output_bert_embeddings(domain_types, data_size):
    for domain in domain_types:
        _ = tokenize_encode_bert_sentences(tokenizer_d, model_d, list(X_dict["X_train_" + domain][:data_size]),
                                           data_path + "all_bert/" + "encoded_" + domain + "_train_" +
                                           str(data_size))
        _ = tokenize_encode_bert_sentences(tokenizer_d, model_d, list(X_dict["X_train_" + domain][:data_size]),
                                           data_path + "all_bert/" + "encoded_" + domain + "_dev_" + str(data_size))


if __name__ == '__main__':
    data_path = "../data/"
    tokenizer_d = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_d = DistilBertModel.from_pretrained('distilbert-base-uncased')
    domains = ["tw", "az", "mv", "fi"]
    data_size = 3000

    X_dict = load_np_files(data_path=data_path, domains=domains, data_types=["train", "dev"], load_feature=True)
    output_bert_embeddings(domains, data_size)
