# domain-adaptation-nlp

## Dataset
Our amazon dataset can be downloaded [here](https://drive.google.com/file/d/1zq_ltCCvozTCrdGReCefkhgZR-ueszda/view?usp=sharing).
Put this file in a folder called "[root]/data/amazon_reviews".

This data contains 2000 samples of the four categories in the amazon reviews data:
* Books
* Electronics
* Home and Kitchen
* Movies and TV



We choose these categories because they are frequently used in nlp sentiment analysis domain adaptation papers.

Our movie dataset can be downloaded [here](https://drive.google.com/file/d/199vK4As0u8MR6_BVXkNqRS2wXcBgA5Jo/view?usp=sharing)
The original dataset is [here](https://ai.stanford.edu/~amaas/data/sentiment/).

You can open the data (for example the amazon data) using the following code, 
although this step should be already included in any function you need to run.

```
with open("../data/amazon_reviews/amazon_4.pickle", "rb") as fr:
        all_data = pickle.load(fr)
```

For each element in the amazon data, and for the movie data, the structure is as follows:
* [0] bert embeddings ([CLS] layer)
* [1] y labels (0 means negative and 1 means positive)
* [2] domain name


## Balanced Conf Model and Few Labels Models
Run sentiment_classification_amazon.py from the root directory.

## Householder Transformation
[To do]

