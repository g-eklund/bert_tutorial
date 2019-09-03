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



