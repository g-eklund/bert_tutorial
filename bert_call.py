

from bert-serving.client import BertClient

bc = BertClient()
strings = ['hej hopp', 'hej hopp']
enc = bc.encode(strings)

