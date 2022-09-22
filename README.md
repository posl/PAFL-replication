PAFL: Probabilistic Automaton-based Fault Localization for RNNs
====
This is the replication package of PAFL (Probabilistic Automaton-based Fault Localization for RNNs).

We use docker and docker-compose to prepare two containers, one for the main processing of PAFL and the other for prism (probabilistic model checking), and link them by docker-compose.

We have tested to work with M1 Macbook Pro BigSur version 11.2.3.

Prism is used to get the state transitions of PFAs as logs.

Our experimental results can be reproduced following steps:
1. Containers preparation
2. PAFL processing
3. Experiments

*require: make, docker, docker-compose*

# How to run
Current directory after clone this repository:
```bash
$ pwd
# /XXXX/YYYY/PAFL-replication
```

## 0. Download datasets and trained models
Download the directory from google drive of the following link and put it under `PAFL-relplicatioin/data`. \
https://drive.google.com/drive/folders/1EcW-BU75erBuTC-cWFmuKo36cUiysP5s?usp=sharing

You can see two directories (`trained_models` and `training_data`) in the link:
- `trained_models` contains the pkl files of the models trained on each dataset. We conducted bootstrap sampling and trained models with these bootstrap samle, so each directory of the dataset contains multiple model files.
- `training_data` contains datasets used for training, word embedding matrices, and pre- and post-split datasets.

After that, `PAFL-relplicatioin/data/trained_models` and `PAFL-relplicatioin/data/training_data` should be existed.

## 1. Containers preparation
Build docker images:
```bash
$ make b
```

Run docker containers:
```bash
$ make u
```
At this point, two containers, `pafl_server` and `prism_server`, should be running.

## 2. PAFL processing
Enter the PAFL container:
```bash
$ make cmain
```

Extract PFAs for all settings, datasets:
```bash
$ cd PAFL/extract_pfa
$ bash extract_pfa.sh  
```
- Extracted PFAs (pm file) is saved in `data/pfa_model/{variant}/{dataset}/boot_{bootid}/k={k}/train_XXXXXX.pm`.
  - variant: srnn, gru, lstm
  - dataset: tomita_3, tomita_4, tomita_7, mr, imdb
  - bootid: 0 to 9
  - k: 2,4,6,8,10

Get PFA state transitions and prediction results (PFA spectrum):
```bash
$ cd ../get_pfa_spectrum
$ bash get_pfa_spectrum.sh
```
- PFA spectrum are saved in `data/pfa_model/{variant}/{dataset}/boot_{bootid}/k={k}/val/`.
- They are recorded in a txt file with state transitions for successful and failed predictions.

Calculate suspicious scores:
```bash
$ cd ../get_suspiciousness
$ bash do_calc_ngram_susp.sh
```
- The top20 suspicious scores for each n-gram are recorded in `data/pfa_susp/pfa_ngram_susp/{variant}/{dataset}/boot_{boot_id}/k={k}/n={n}/val/ochiai_susp_top20.json`.

Extract data samples by PAFL:
```bash
$ cd ../extract_data
$ bash make_pfa_trace.sh
$ bash extract_data.sh
```
- Extracted data samples are saved in `data/extracted_data/{variant}/{dataset}/boot_{boot_id}/k={k}/n={n}/theta={theta}/ex_data.pkl`.

## 3. Experiments
First, move the experiments directory:
```bash
$ cd ../../experiments
$ pwd
# /src/experiments
```

### RQ1
Reproduce RQ1 results:
```bash
$ cd rq1
$ bash rq1.sh
```
- The box plot of Average ANS scores or each n and dataset (same as Fig.8 in our paper) is saved in `/src/experiments/rq1/rq1_figs/rq1_{variant}_{method}.pdf`.
- `method`: ample, ochiai, ochiai2, tarantula, dstar

### RQ2
#### vs. random
Reproduce RQ2 results (vs. random):
```bash
$ cd ../rq2-vsrandom
$ python compare_random.py
$ python barplot_n.py
```
- After executing `compare_random.py`, the csv files are saved in `Tab6-avg.csv`, `Tab6-stat.csv`, `Tab7-avg.csv`, `Tab7-stat.csv` (same tables as Table 6, 7 in our paper).
- After executing `barplot_n.py`, the pdf file is saved in `Fig11-barplot-n.pdf` (same as Fig.11 in our paper).
#### vs. SOTA FL
Reproduce RQ2 results (vs. SOTA FL):
```bash
$ cd ../rq2-vssota
$ bash run_rnnrepair.sh
```
- By executing `run_rnnrepair.sh`, you can get the results of applying RNNRepair.
- Other results can be gotten in the notebooks in the same directory.

### Discussion2
Reproduce Discussion2 results:
```bash
$ cd ../dis2_kn
$ bash disc2.sh
```
- After executing above, we can check the full results of Discussion2 in `./discussion1_figs/{model_type}_{method}_kn.pdf`.

### Discussion3
Reproduce Discussion3 results:
```bash
$ cd ../dis3_tie
$ bash run_ties.sh
```
- After executing above, we can check the results of Discussion3 in `./tie_rate_per_k_{model_type}.csv`. These tables are same as Table 11 in our paper.

### Discussion4
Reproduce Discussion4 results:
```bash
$ cd ../dis4_dfa
$ bash run_dfavspfa.sh
```
- After executing above, we can check the results of Discussion4 in `./Fig12.pdf`. The figure is same as Figure12 in our paper.
