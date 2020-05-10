# Directory Structure
```text
.
├── data
│   ├── r8-test-all-terms.txt
│   └── r8-train-all-terms.txt
├── embeddings
│   ├── glove.6B.50d.txt
│   └── GoogleNews-vectors-negative300.bin
├── README.md
├── run.py
└── utils
    ├── data.py
    ├── embeddings.py
    └── __init__.py

3 directories, 9 files
```
# Command Line Interface
```text
usage: run.py [-h] --embedding {glove,word2vec}

optional arguments:
  -h, --help            show this help message and exit
  --embedding {glove,word2vec}
                        Glove Embedding or Word2Vec Embedding
usage: run.py [-h] --embedding {glove,word2vec}
run.py: error: the following arguments are required: --embedding
```


# Data - Reuters-21578 R8

```text
fetch data 
```
[](https://www.cs.umb.edu/~smimarog/textmining/datasets/)

```text
features: sentenes
```
```text
labels: acq, crude, earn, grain, interest, money-fx, ship, trade	
```

# Embeddings - Glove,Word2Vec

## Word2Vec
```text
fetch embedding
```
[Word2Vec](https://code.google.com/archive/p/word2vec/)
## Glove
```text
fetch embedding
```
[GloVe](http://nlp.stanford.edu/data/glove.6B.zip)
[Direct Link](https://nlp.stanford.edu/projects/glove/)

```shell
gunzip GoogleNews-vectors-negative300.bin.gz
```
# Train the RandomForestClassifier model with pre-trained Glove Embedding

## CLI
```shell
 python run.py --embedding glove
````
```text
not found word sentence vectors 0/5485
not found word sentence vectors 0/2189
```
## Accuracy Score
```text
train score 0.9992707383773929 
```
```text
test score 0.9328460484239379 
```

# Under The Maintenance...