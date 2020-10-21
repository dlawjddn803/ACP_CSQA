# I Know What You Asked: Graph Path Learning using AMR for Commonsense Reasoning

Code for our COLING2020 paper



## 0. Environment Setup

The code runs with python 3.6.
All dependencies are listed in [requirements.txt](requirements.txt)

`pip install -r requirements.txt`


## 1. Data Preparation 

### 1.1 Question only, concept information extraction

Download QA dataset in [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) and place it in the folder name 'dataset'

Download Artifacts:
```
./scripts/download_artifacts.sh
```
Download Preprocessed ConceptNet in [here]() and place it in conceptnet folder 

Make QA dataset have fake AMR dataset form (Please check before you run the code)
```
cd dataset
python preprocess.py
```
Then, place train.txt, dev.txt, test.txt set into ./dataset/csqa/train, ./dataset/csqa/dev, ./dataset/csqa/test folder respectively


Prepare train/dev/test data:
```
cd ..
./scripts/prepare_data.sh -v 2 -p [project_path]
```

We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**) for tokenizing.

First, start a CoreNLP server.

Then, annotate QA sentences:
```bash
./scripts/annotate_features.sh amr_data/amr_2.0/csqa
```

Data Preprocessing
```bash
./scripts/preprocess_2.0.sh
```

Then, Make QA dataset sentences' AMR data using [stog](https://github.com/sheng-z/stog)'s pretrained model.

Before u run this code, please make sure you modify config file. 
 
```bash
python -u -m stog.commands.predict \
    --archive-file ckpt-amr-2.0 \
    --weights-file ckpt-amr-2.0/best.th \
    --input-file data/AMR/amr_2.0/train.txt.features.preproc \
    --batch-size 2 \
    --use-dataset-reader \
    --cuda-device 0 \
    --output-file amr_data/amr_2.0/train.pred.txt \
    --silent \
    --beam-size 5 \
    --predictor STOG
```

Then, prepare vocab dataset


```bash
sh ./scrips/prepare.sh
```

### 1.2 Let's divide our data into a bathches to attach our knowledge graph.
It will takes some time as we write all the paths of the ACP graph. You will need enoguh space to save the data. (mnt folder would be fine choice)
```bash
cd prepare
python generate_batch.py 50 10 10 # train/dev/test
python generate_prepare.py generate_bash 50 10 10 AMR_CN_PRUNE 
sh cmd_extract_train.sh
sh cmd_extract_dev.sh
```

### 1.3 Then Let's combine divided file
```
python generate_prepare.py combine 50 10 10 train/dev/test
```

### 1.4 Make in-house dataset
```
python divide_inhouse_data.py
```

## 2. Train
Train our model
```
sh train.sh
``` 

## 3. Evaluate
Evaluate our model
```
sh evaluate.sh
``` 

## Acknowledgement
We adopted some modules or code snippets from AllenNLP, sheng-z/stog, jcyk/gtos. Thanks to these open-source projects!

## Contact
For any questions, please send me an email to Jungwoo Lim(wjddn803@korea.ac.kr)
