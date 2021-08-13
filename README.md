# Outline for README

- Package Overview
- Run Scripts
- Requirements

## Package Overview

```
├─data
│  ├─source
│  │  ├─aac
│  │  │      negative_aac.csv
│  │  │      positive_aac.csv
│  │  └─fasta
│  │         negative.fasta
│  │         positive.fasta
│  └─train
│          data.csv
│          label.csv
│          merge.csv
│  model.py
│  nn.py
│  out.txt
```

- model.py
  - the script used for training samples with SVM, KNN and RF (random forest)
- nn.py
  - the script used for training samples with Neural Network
- data/source/fasta
  - raw FASTA file
- data/source/aac
  - the Amino Acid Composition (AAC) calculates the frequency of each amino acid type in a protein or peptide sequence
- data/train
  - data.csv -- merge positive_aac.csv and negative_aac.csv, used for the script model.py
  - label.csv -- mark the corresponding positive and negative samples, used for the script model.py
  - merge.csv -- merge data.csv and label.csv, used for the script nn.py
- out.txt
  - the results we got

## Run Scripts

### model.py

```
python model.py -e ESTIMATOR [-dp DATA_PATH] [-lp LABEL_PATH]
```

- -e, **required**, the estimator will be used, include **['svm', 'knn', 'rf']**
- -dp, the path of data.csv
- -lp, the path of label.csv

### nn.py

```
python nn.py [-e EPOCHS] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-dp DATA_PATH]
```

- -e, epochs, default=30
- -bs, batch_size, default=64
- -lr, learning_rate, default=0.01
- -dp, the path of merge.csv

## Requirements

In the experiment, all scripts run in python 3.7, and the third-party packages we used are listed below: 

```
numpy		1.16.2
pandas		0.25.0
sklearn		0.23.2
torch		1.4.0+cpu
torchvision	0.5.0+cpu
```

