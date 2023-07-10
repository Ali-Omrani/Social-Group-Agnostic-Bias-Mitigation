# Replication

Code for the Experiments on Static Word Embeddings

# Environment Setup

1. To setup the required Python packages, run `pip install -r requirements.txt`

2. Download and Unzip [Wikitext-103 data](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)

3. Train Word2Vec in Python

```
from gensim.models import Word2Vec
embed = Word2Vec(corpus_file=<path to corpus>, vector_size=300, workers=48,
                    window=5, min_count=5, epochs=5)
embed.save("./data/embedding/31.model")
```

In order to run experiments on other embeddings, change the `sgns_wikitext_path` varibale in `config.py`.

# Experiment 1 - Bias Reduction

`notebooks/Paper -   Experiment 1 - Bias reduction.ipynb` contains the code for repeated bias reduction evaluation (ECT/EQT) of SCM-based debiasing to social-gorup-specific debiasing.

# Experiment 2 - Word Embedding Utility

`notebooks/Paper -   Paper - Experiment 2 - Embedding Quality.ipynb` contains the code for evaluating the utility of SCM-based debiasing and simultaneous social-group-specific debiasing for gender, race, and age.
