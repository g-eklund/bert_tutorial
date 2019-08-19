#!/bin/bash
virtualenv venv -p python3
source venv/bin/activate

wget 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
mkdir bert 
unzip 'uncased_L-12_H-768_A-12.zip' -d bert
rm 'uncased_L-12_H-768_A-12.zip'

wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
mkdir fasttext 
unzip 'wiki-news-300d-1M.vec.zip' -d fasttext
rm 'wiki-news-300d-1M.vec.zip'

pip install -U jupyter tensorflow==1.14.0 gensim bert-serving-server bert-serving-client
bert-serving-start -model_dir '/home/gurra/Code/bert/bert/' -num_worker=4 
