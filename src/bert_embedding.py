from transformers import DistilBertTokenizer, DistilBertModel
from utils import *


def output_bert_embeddings(domain_types, data_size):
    for domain in domain_types:
        _ = tokenize_encode_bert_sentences(tokenizer_d, model_d, list(X_dict["X_train_" + domain][:data_size]),
                                           data_path + "all_bert/" + "encoded_" + domain + "_train_" +
                                           str(data_size))
        _ = tokenize_encode_bert_sentences(tokenizer_d, model_d, list(X_dict["X_train_" + domain][:data_size]),
                                           data_path + "all_bert/" + "encoded_" + domain + "_dev_" + str(data_size))


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


if __name__ == '__main__':
    data_path = "../data/"
    tokenizer_d = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_d = DistilBertModel.from_pretrained('distilbert-base-uncased')
    domains = ["tw", "az", "mv", "fi"]
    data_size = 3000

    X_dict = load_np_files(data_path=data_path, domains=domains, data_types=["train", "dev"], load_feature=True)
    output_bert_embeddings(domains, data_size)
