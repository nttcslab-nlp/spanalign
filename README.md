# SpanAlign: Sentence Alignment Method based on Cross-Language Span Prediction and ILP
This repository includes the software described in "[SpanAlign: Sentence Alignment Method based on Cross-Language Span Prediction and ILP](https://www.aclweb.org/anthology/2020.coling-main.418/)" published at COLING'20.

## Setup
This software is tested on the following.
* GeForce RTX 2080Ti
* Python 3.8.6
* CUDA 10.1
* ILOG CLPEX 12.8.0.0
* torch 1.7.1+cu101
* transformers 4.1.1
* nltk
* tqdm 4.54.1
* h5py 2.10.0
* sentencepiece 0.1.94
* tensorboardX 2.1
* tabulate 0.8.7

These python libraries may be installed by using pip as follows:
```sh
pip install -r requirements.txt
```

## How to use with sample data
### Preprocessing
First, you need to convert dataset to the json format of SQuAD 2.0 QA task. (For this example, we create train/dev/test json files from one sample data.)
```sh
# Preprocessing for train/development sets
python scripts/pairDoc2SQuAD.py data/sample.{pair,l1,l2} data/train.json train
python scripts/pairDoc2SQuAD.py data/sample.{pair,l1,l2} data/dev.json dev
# Preprocessing for a test set
python scripts/doc_to_overlaped_squad.py data/sample.{l1,l2} data/test.json -t test
```

When you have a lot of json files of training/development set, you can separately convert these files and merge them into one file by using `scripts/mergeSQuADjson.py`, like this:
```sh
python scripts/mergeSQuADjson.py -s data/train*.json --squad_version 2.0 data/train.json
```

### Fine-tuning Model and Get Alignments
According to your environment, You need to rewrite `__SET_DIR_PATH__` to `the root path of this repository` in `experiments/run_{finetuning,extract}.sh` and `__SET_CPLEX_PATH__` to `the executable path of CPLEX` in `scripts/get_sent_align_for_overlap.py`.
These commands will fine-tune XLM-RoBERTa for sentence alignment.
```sh
cd experiments
sh ./run_finetuning.sh
```

The following script `run_extract.sh` do the cross-langauge span prediction for extracting alignment hypothesises and optimize these alignments by using Integer Linear Programming.
```sh
sh ./run_extract.sh ./finetuning ./output ../data/test.json test_sample
```

The alignments results with three symmetization methods are at `experiments/output/test/test.{e2f,f2e,bidi}.pair`.
```shell
$ cat output/test/test.bidi.pair
[]:[1]:1.0000
[12,13]:[2,3,4,5,6,7,8,9,10,11,12,13,14]:7.0965
[1]:[]:1.0000
[2,3,4,5,6,7,8,9,10,11]:[15]:5.1970
```

Here, `bidi` means bi-directional symmetization in our paper.

### Evaluate
Sentence Alignment accuracies can be calculated as follows.
```sh
$ python ../scripts/score.py -g ../data/sample.pair -t ./output/test/test.bidi.pair
 ---------------------------------
|             |  Strict |    Lax  |
| Precision   |   0.000 |   0.500 |
| Recall      |   0.000 |   0.250 |
| F1          |   0.000 |   0.333 |
 ---------------------------------
  trg/src  0                      1
---------  ---------------------  ----------------------
        0  0.000/0.000/0.000 (0)  0.000/0.000/0.000 (0)
        1  0.000/0.000/0.000 (0)  0.000/0.000/0.000 (11)
        2  0.000/0.000/0.000 (0)  0.000/0.000/0.000 (0)
        3  0.000/0.000/0.000 (0)  0.000/0.000/0.000 (1)
```

## License
This software is released under the NTT License, see `LICENSE.txt`.
