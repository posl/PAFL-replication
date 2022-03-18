PAFL: Probabilistic Automaton-based Fault Localization for RNNs
====
This is the replication package of PAFL (Probabilistic Automaton-based Fault Localization for RNNs).

Prepare two containers, one for the main processing of PAFL and the other for prism (probabilistic model checking), and link them by docker-compose.

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

Download the directory from google drive of the following link and put it under `PAFL-relplicatioin/data`. \
https://drive.google.com/drive/folders/1EcW-BU75erBuTC-cWFmuKo36cUiysP5s?usp=sharing

## 1. Containers preparation
Build docker images:
```bash
$ make b
```

Run docker containers:
```bash
$ make u
```

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
- Extracted PFAs (pm file) is saved in `data/pfa_model/lstm/{dataset}/boot_{bootid}/k={k}/train_XXXXXX.pm`.
  - dataset: tomita_3, tomita_4, tomita_7, mr, imdb
  - bootid: 0 to 9
  - k: 2,4,6,8,10

Get PFA state transitions and prediction results (PFA spectrum):
```bash
$ cd ../get_pfa_spectrum
$ bash get_pfa_spectrum.sh
```
- PFA spectrum are saved in `data/pfa_model/lstm/{dataset}/boot_{bootid}/k={k}/val/`.
- They are recorded in a txt file with state transitions for successful and failed predictions.

Calculate suspicious scores:
```bash
$ cd ../get_suspiciousness
$ python do_calc_ngram_susp.py
```
- The top20 suspicious scores for each n-gram are recorded in `data/pfa_susp/pfa_ngram_susp/lstm/{dataset}/boot_{boot_id}/k={k}/n={n}/val/ochiai_susp_top20.json`.

Extract data samples by PAFL:
```bash
$ cd ../extract_data
$ bash make_pfa_trace.sh
$ bash extract_data.sh
```
- Extracted data samples are saved in `data/extracted_data/lstm/{dataset}/boot_{boot_id}/k={k}/n={n}/theta={theta}/ex_data.pkl`.

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
$ cd RQ1
$ python rq1_boxplot.py
```
- The box plot of Average ANS scores or each n and dataset (same as Fig.7 in our paper) is saved in `/src/experiments/RQ1/rq1.pdf`.

### RQ2
Reproduce RQ2 results:
```bash
$ cd ../RQ2
$ bash pred_data_samples.sh
$ bash utest.sh
```
- The prediction performances on PAFL-selected (ex) / randomly selected (rand) data samples are saved as csv files in `data/extracted_data/pred_table_for_look/boot_{boot_id}/n={n}_theta={theta}.csv`.
- These prediction performances are averaged and compared between ex and rand, and the counts of cases where ex has lower prediction performance are saved as a csv file in `experimements/RQ2/pred_results/n={n}_theta={theta}.csv` (same as Table 5 in our paper).
- The results of utest (p-value and effect size) are saved in `experimements/RQ2/stats_results/raw/n={n}_theta={theta}.csv`.
- The counts of statistically significant and large effect sizes for ex are saved in `experimements/RQ2/stats_results/cnt/n={n}_theta={theta}.csv` (same as Table 6 in our paper).
