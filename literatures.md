# Literatures
## NER
* [BERT(please read NER example in section 5.3)](https://arxiv.org/pdf/1810.04805.pdf)
## Domain alignment
* [Zero shot NER](https://arxiv.org/pdf/2002.05923.pdf)
  * the closest task with ours
* [Cross Domain NER(SOTA)](https://www.aclweb.org/anthology/P19-1236.pdf)

## Context-Specific Word Embedding
* [Elmo](https://arxiv.org/pdf/1802.05365.pdf)

# Tutorials
* [NER with BERT](https://medium.com/@yingbiao/ner-with-bert-in-action-936ff275bc73#:~:text=NER%20is%20a%20task%20in,model%20for%20NER%20downstream%20task.)

# NER Dataset
* [CoNLL2003](https://huggingface.co/datasets/conll2003)
  [Alternative link](https://www.clips.uantwerpen.be/conll2003/ner/)
  [Alternative link](https://github.com/glample/tagger/blob/master/dataset/eng.testa)
* [CrossNER](https://zihanliu1026.medium.com/crossner-evaluating-cross-domain-named-entity-recognition-1a3ee2c1c42b)
  * need to edit the categories
* [Cross Domain NER(SOTA)](https://github.com/jiachenwestlake/Cross-Domain_NER)
  * use CoNLL-2003 as source, use CBS SciTech News as target
* [Other NER](https://github.com/juand-r/entity-recognition-datasets)
  * used wikigold and SEC-filings

# Classification Dataset
* [20-Newsgroups](https://dl.acm.org/doi/abs/10.1145/1281192.1281218?casa_token=jMMh4etuT_cAAAAA%3A83HBb43uGWalKdDbRZj3UFZy7JG3fDkt10kUDpGBI_GWluphIp9tiXbe4YZPZo-uvfuOZ_9kw-K04g)
* [Amazon Data](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/)
  * Usually use reduced set(Books, DVDs, Electronicsand Kitchen)(Blitzer et al.)
* [SentDat](https://dl.acm.org/doi/pdf/10.1145/1772690.1772767)
  * (video game(V), electronics(E), software(S) and hotel (H))

# Multilingual Dataset
* [NTCIR](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.364.9258&rep=rep1&type=pdf)
  * EN & CN
* [Sanders/GermEval](https://aclanthology.org/L18-1101.pdf)
  * EN & German, sentiment classification, pos/neu/neg, multilingual-multiclass




# pretrained BERT
* [CoNLL2003 pretrained BERT](https://huggingface.co/dslim/bert-base-NER)
