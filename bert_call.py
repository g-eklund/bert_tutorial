

from bert-serving.client import BertClient
import numpy as np

bc = BertClient()
strings = ['hej hopp', 'hej hopp']
enc = bc.encode(strings)


def cosine_sim(s1, s2):
    return np.dot(s1[0,:],s2[0,:])/(np.linalg.norm(s1)*np.linalg.norm(s2)+1e-9)
