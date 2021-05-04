# domain-adaptation-nlp

## Dataset
Our amazon dataset can be downloaded [here](https://drive.google.com/file/d/1zq_ltCCvozTCrdGReCefkhgZR-ueszda/view?usp=sharing).
Put this file in a folder called "data/amazon_reviews".

Our movie dataset can be downloaded [here](TOADD)

This data contains 2000 samples of the four categories in the amazon reviews data:
* Books
* Electronics
* Home and Kitchen
* Movies and TV

For each domain in the data, it contains three items: 
* [0] bert embeddings ([CLS] layer)
* [1] y labels (1 means negative and 2 means positive)
* [2] domain name

We choose these categories because they are frequently used in nlp sentiment analysis domain adaptation papers.

You can open this file using the following code, although this step is already included in any function you need to run.

```
with open("../data/amazon_reviews/amazon_4.pickle", "rb") as fr:
        all_data = pickle.load(fr)
```
    
