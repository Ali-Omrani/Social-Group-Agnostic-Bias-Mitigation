# Replication

Code for the Experiments on Contextualized Word Embedding.

We considered **DPCE** as a fine-tuning aproach, and **ADEPT** as a prompt-tuning approach, and conducted experiments on the [**bert-large-uncased**](https://huggingface.co/bert-large-uncased) pre-trained model from [HuggingFace](https://huggingface.co/).

We provide bash scripts and codes to replicate our findings.

# Environment Setup

Our environment is:

- Ubuntu servers with NVIDIA GeForce RTX 3090-ti (24G) GPUs
- cuda 11.2
- packages with certain versions

### Create environment:

```bash
conda create -n debias python=3.10.8
conda activate debias
```

### Install pytorch and python packages:

```bash
pip install -r requirements.txt
```

# Data

We've already included word lists for attributes in the `./data` folder, so there is no need to acquire them from other resources. As for larger corpora, you can download News-Commentary v15 [here](https://data.statmt.org/news-commentary/v15/documents.tgz) and Hugging Face's BookCorpus replica [here](https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2). New-Commentary alone can support gender debiasing. You may need to create a new text file by combining the two corpora mentioned above so that there are sufficient sentences for religion and scm debiasing.

# Debias Models

Variablaes:

- Algorithm: **ADEPT** or **DPCE**.

- Bias: **gender** (group-specific), **religion** (group-specific), or **scm** (group-agnostic).

### Collect sentences:

```bash
cd script
sh collect_sentences.sh bert [corpus_path] [bias] final
```

### Debias:

```bash
sh debias.sh bert 0 [algorithm] [bias] 0
```

# Benchmarks

```bash
cd bias-bench/experiments
```

### GLUE benchmarks

```bash
sh DPCE_glue.sh [output_folder] [model_name_or_path] # for DPCE
```

```bash
sh run_adept_glue.sh [output_folder] [model_name_or_path] # for ADEPT
```

### Bias benchmarks

```bash
sh DPCE_all.sh [model_name_or_path] # for DPCE
```

```bash
sh run_adept_crows.sh [model_name_or_path] # for ADEPT
sh run_adept_seat.sh [model_name_or_path]  # for ADEPT
sh run_adept_seat.sh [model_name_or_path]  # for ADEPT

```
