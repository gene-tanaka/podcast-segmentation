from wiki_loader import WikipediaDataSet
import io

'''
The following function was taken from https://fasttext.cc/docs/en/english-vectors.html
'''
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

word2vecModel = load_vectors('wiki-news-300d-1M-subword.vec')

# class SegmentationModel(self):
# 	def __init__(self, init_size):
		

train_dataset = WikipediaDataSet('./data', word2vec=word2vecModel, folder=True)