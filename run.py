from wiki_loader import WikipediaDataSet
import io
import torch

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


		

train_dataset = WikipediaDataSet('./data', word2vec=word2vecModel, folder=True)

class SentenceRepresentation(torch.nn.Module):
	def __init__(self, init_size, hidden_size):
		super(SegmentationModel, self).__init__()
		self.init_size = init_size
		#could also consider adding dropout to 
		self.firstLayer = torch.nn.LSTM(input_size=input_size, 
										hidden_size=hidden_size, 
										num_layers=2, 
										bidirectional=True)
	def forward(self, input_x):
		output = self.firstLayer(input_x)
		#not sure what batch_size should be.
		maxes = self.max_pooling(output, batch_size)
		return maxes

	def max_pooling(self, output, batch_size):
		'''This function was adapted from https://github.com/koomri/text-segmentation/blob/874d6ef3ca0e402709b70924608d0894be9a93e1/models/max_sentence_embedding.py
		'''
		maxes = Variable((torch.zeros(batch_size, output.size(2))))
        for i in range(batch_size):
            maxes[i, :] = torch.max(output[:lengths[i], i, :], 0)[0]

        return maxes

class SecondNN(torch.nn.Module):
	def __init__(self, init_size, hidden_size):
		super(secondNN, self).__init__()
		self.init_size = init_size
		#could also consider adding dropout to 
		self.secondLayer = torch.nn.LSTM(input_size=input_size, 
										hidden_size=hidden_size, 
										num_layers=2, 
										bidirectional=True)
	def forward(self, input_x):
		outputs = self.secondLayer(input_x)
		softmax = torch.nn.Softmax()
		scores = softmax(output)
		return scores

class LastLayer(torch.nn.Module):
	def __init__(self, init_size, hidden_size):
		#not super sure what the parameters here should be
		self.fullyConnected = torch.nn.Linear(init_size, init_size)
	
	def forward(self, input_x):
		outputs = self.fullyConnected(input_x)
		softmax = torch.nn.Softmax()
		scores = softmax(output)
		return scores










