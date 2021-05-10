# domain-adaptation-nlp

## Dataset
Our amazon dataset can be downloaded [here](https://drive.google.com/file/d/1zq_ltCCvozTCrdGReCefkhgZR-ueszda/view?usp=sharing).
Put this file in a folder called "[root]/data/amazon_reviews".

This data contains 2000 samples of the four categories in the amazon reviews data:
* Books
* Electronics
* Home and Kitchen (Kitchen)
* Movies and TV (DVDs)

We choose these categories because they are frequently used in nlp sentiment analysis domain adaptation papers.

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


## Instructions to run
### Balanced Conf Model and Few Labels Models
* Create an output folder under this root directory if it does not exist.
* Run sentiment_classification_amazon.py from the root directory.

### Householder Transformation
* Adjust the n of n_fold want to use(default: 1000).
* Run domain_space_alignment.py from the root directory.
