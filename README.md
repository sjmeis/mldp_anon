# *A Comparative Analysis of Word-Level Metric Differential Privacy: Benchmarking The Privacy-Utility Trade-off*
## Replication Package
In this directory, we provide the full replication package required to reproduce the experiments described in our paper. You will find the following files:

- `perturb.ipynb`: base file for perturbing text data using our seven selected mechanisms
- `train.ipynb`: used to train models on the perturbed data (and achieve accuracy scores)
- `pd_metrics.ipynb`: for calculating N_w and S_w, as described in the paper
- `calculate_privacy_stats.ipynb`: for calculating the other described privacy metrics
- `util/algorithms.py`: code for all mechanism implementations
- `util/train.py`: util for model traning using keras
- `util/wordvec_load.py`: util for loading GloVe embeddings

Data files of note include:

- `Data/imdb_preprocessed_train.csv`
- `Data/imdb_preprocessed_test.csv`
- `Data/ag_news_preprocessed_train.csv`
- `Data/ag_news_preprocessed_test.csv`

These are the (preprocessed) text files used to train our models for benchmarking.

NOTE: the required GloVe embedding files are not included in this package due to size. These can be downloaded from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/). Also note that you will have to  enter the path to these word embedding files manually in the above-mentioned notebooks (marked in the code by comments).

In order to install all Python dependencies, please run: `pip install -r requirements.txt`

The full code repository will be published on Github following acceptance.