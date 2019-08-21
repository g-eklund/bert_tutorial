%sh 
/databricks/python3/bin/pip install fasttext gensim 
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
rm wiki-news-300d-1M.vec.zip

from gensim.models import KeyedVectors
import re
import numpy as np
filename = 'wiki-news-300d-1M.vec'
fasttext_model = KeyedVectors.load_word2vec_format(filename)




def get_fasttext_wv(word, model, vec_len=300):
    try:
        wv = model[word]
    except:
        wv = np.zeros(vec_len)
    return wv
  
  
def doc_similarity(a, b):
    return np.dot(a, b)/ (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) 
  
  
def fasttext_doc2vec(string, model, stop_words=list(), weights=dict(), smoothing_factor=1, n_dim=300):
    """
    sentence embedding from a word embedding model and string sentence. 
    args:
    string(str): the string to be embedded
    model(gensim.models.keyedvectors.Word2VecKeyedVectors): the model
    stop_words(list): optional stop words to be removed
    weights(dict): optional dict with weights of words 
    smoothing_factor(int): helps smoothing the doc2 vec. A nonzero value is as 
    if all words without an explicit weight would have that value as explict. A smoothing factor=0 
    is equivalent to ignoring words that has no explcit weight. Set to 1 if unsure.
    
    """
    words = set(re.findall("[\w\d'-]+", string.lower())) # ignores multiple uses of a word in the doc
    
    word_weights = []
    if words:
        word_vectors = [get_fasttext_wv(word, model) for word in words if word not in stop_words]
        for word in words:
            try:
                word_weights.append(weights[word])
            except:
                word_weights.append(smoothing_factor)
                
        se = np.mean([vec / (np.linalg.norm(vec) + 1e-9) * weight for vec, weight in zip(word_vectors, word_weights)], axis = 0)
    else:
        print('Zero doc2vec vector for: ' + string)
        se = np.zeros(n_dim)
    return se


string1 = 'the quick brown fox'
string2 = 'jumped over the lazy dog'

str_1_emb = fasttext_doc2vec(string1, fasttext_model)
str_2_emb = fasttext_doc2vec(string2, fasttext_model)

doc_similarity(str_1_emb, str_2_emb)

---
%sh 
/databricks/python3/bin/pip install bert-embedding

---

from bert_embedding import BertEmbedding

bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
 Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
 As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
BERT is conceptually simple and empirically powerful. 
It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
sentences = bert_abstract.split('\n')
bert_embedding = BertEmbedding()
result = bert_embedding(sentences)



bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')

string1 = 'the quick brown fox'
string2 = 'jumped over the lazy dog'

s1 = bert_embedding([string1])
s2 = bert_embedding([string2])



def bert_sentence_embedding(emb):
    """very simple embedding, just averaging the tokens 
    
    """
    embs=list()# = np.array()
    for i, t in enumerate(emb[0][0]):
        print(i, t)
        embs.append([emb[0][1][i]])
    
    return np.mean(np.array(embs).reshape(i+1,-1), axis =0)
    
embs = bert_sentence_embedding(s1)



