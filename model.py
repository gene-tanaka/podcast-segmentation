import torch

class Model(nn.Module):
	def __init__(self, hidden=128, num_layers=2):
		super(Model, self).__init__()
		self.sentenceRepresentation = SentenceRepresentation(300, 256)
		self.secondNN = torch.nn.LSTM(input_size=self.SentenceRepresentation.hidden *2, 
										hidden_size=hidden, 
										num_layers=num_layers, 
										dropout=0,
										bidirectional=True)
		self.lastLayer = torch.nn.Linear(hidden *2, 2)
		self.loss = torch.nn.CrossEntropyLoss()
	def forward(self, batch):
		encoded_batch = self.sentenceRepresentation(batch)
		secondNN_output = self.secondNN(encoded_batch)
		x = self.lastLayer(secondNN_output)
		return x


class SentenceRepresentation(torch.nn.Module):
	def __init__(self, init_size, hidden_size):
		super(SegmentationModel, self).__init__()
		self.init_size = init_size
		self.hidden = hidden_size
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
