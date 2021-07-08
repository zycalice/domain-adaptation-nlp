import json
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.utils import *


def output_bert_embeddings(domain_types, data_size, X_dict):
    for domain in domain_types:
        _ = tokenize_encode_bert_sentences_batch(tokenizer_d, model_d, list(X_dict["X_train_" + domain][:data_size]),
                                                 data_path + "all_bert/" + "encoded_" + domain + "_train_" +
                                                 str(data_size))
        _ = tokenize_encode_bert_sentences_batch(tokenizer_d, model_d, list(X_dict["X_train_" + domain][:data_size]),
                                                 data_path + "all_bert/" + "encoded_" + domain + "_dev_" + str(
                                                     data_size))


def tokenize_encode_bert_sentences_batch(tokenizer, model, input_sentences, output_path, layer_type="CLS"):
    output = np.zeros([len(input_sentences), 768])
    for i, x in enumerate(input_sentences):
        output[i] = tokenize_encode_bert_sentences_sample(tokenizer=tokenizer, model=model,
                                                          input_sentences=[x],
                                                          layer_type=layer_type)
    if output_path is not None:
        np.save(output_path, output)
    return output


def tokenize_encode_bert_sentences_sample(tokenizer, model, input_sentences, layer_type="CLS"):
    encoded_input = tokenizer(input_sentences, return_tensors='pt', truncation=True, padding=True)
    output = model(**encoded_input)[0]
    if layer_type == "CLS":
        output = output[:, 0, :].detach().numpy()
    elif layer_type == "first subtoken":
        output = output[:, 1, :].detach().numpy()
    else:
        output = output.detach().numpy()
    return output


def fine_tune_bert():
    pass


if __name__ == '__main__':
    data_path = "../data/"
    tokenizer_d = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model_d = DistilBertModel.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer_cased = BertTokenizer.from_pretrained('bert-base-cased')
    model_cased = BertModel.from_pretrained('bert-base-cased')
    tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").bert

    # # Sentiment Classification.
    # domains = ["tw", "az", "mv", "fi"]
    # clf_data_size = 3000
    #
    # X_dict = load_np_files(data_path=data_path, domains=domains, data_types=["train", "dev"], load_feature=True)
    # output_bert_embeddings(domains, clf_data_size, X_dict)
    #
    # # NER v1.
    # with open(data_path + "wiki_sec_word2idx.json") as file:
    #     word2idx = json.load(file)
    # words_list = list(word2idx.keys())
    # encoded_ner_corpus = tokenize_encode_bert_sentences_batch(
    #     tokenizer, model, words_list,
    #     "../data/all_bert/bert_uncased_encoded_ner_corpus")
    # cased_encoded_ner_corpus = tokenize_encode_bert_sentences_batch(
    #     tokenizer_cased, model_cased, words_list,
    #     "../data/all_bert/bert_cased_encoded_ner_corpus")

    # # NER v2.
    # with open(data_path + "conll_tech_word2idx.json") as file:
    #     word2idx = json.load(file)
    # words_list = list(word2idx.keys())
    # _ = tokenize_encode_bert_sentences_batch(
    #     tokenizer, model, words_list,
    #     "../data/all_bert/bert_uncased_encoded_ner_corpus_conll")
    # _ = tokenize_encode_bert_sentences_batch(
    #     tokenizer_cased, model_cased, words_list,
    #     "../data/all_bert/bert_cased_encoded_ner_corpus_conll")
    # _ = tokenize_encode_bert_sentences_batch(
    #     tokenizer_ner, model_ner, words_list,
    #     "../data/all_bert/bert_ner_encoded_ner_corpus_conll")

    # NER v3.
    with open(data_path + "conll_tech_word2idx.json") as file:
        word2idx = json.load(file)
    words_list = list(word2idx.keys())
    _ = tokenize_encode_bert_sentences_batch(
        tokenizer_ner, model_ner, words_list,
        "../data/all_bert/bert_ner_first_token_encoded_ner_corpus_conll",
        layer_type="first subtoken"
    )
