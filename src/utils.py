import numpy as np


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


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
