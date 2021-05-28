import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter
import pickle
from bert_embedding import *
from dataset_multi import load_ner_data
from sklearn.model_selection import train_test_split
from src.domain_space_alignment import ht_lr
import sys

# https://medium.com/@yingbiao/ner-with-bert-in-action-936ff275bc73#:~:text=NER%20is%20a%20task%20in,model%20for%20NER%20downstream%20task.
