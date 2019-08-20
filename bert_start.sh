#!/bin/bash
virtualenv venv -p python3
source venv/bin/activate

wget 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
mkdir bert 
unzip 'uncased_L-12_H-768_A-12.zip' -d bert
rm 'uncased_L-12_H-768_A-12.zip'

pip install -U jupyter tensorflow==1.14.0 bert-serving-server bert-serving-client
bert-serving-start -model_dir '/home/gurra/Code/bert/bert/' -num_worker=4 
