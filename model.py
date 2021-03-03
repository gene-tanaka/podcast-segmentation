import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(torch.nn.Module):
	def __init__(self, hidden=128, num_layers=2):
		super(Model, self).__init__()
		self.sentenceRepresentation = SentenceRepresentation(300, 256, 2)
		self.secondNN = torch.nn.LSTM(input_size=self.sentenceRepresentation.hidden *2, 
										hidden_size=hidden, 
										num_layers=num_layers, 
										dropout=0,
										bidirectional=True)
		self.lastLayer = torch.nn.Linear(hidden * 2, 2)
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, batch):
		self.sentenceRepresentation = self.sentenceRepresentation.double()
		encoded_batch, _ = self.sentenceRepresentation(batch)
		# print("encoded_batch: {}".format(encoded_batch.shape))
		self.secondNN = self.secondNN.double()
		secondNN_output, _ = self.secondNN(encoded_batch)
		# print("secondNN_output: {}".format(secondNN_output.shape))
		self.lastLayer = self.lastLayer.double()
		x = self.lastLayer(secondNN_output)
		# print("x: {}".format(x.shape))
		return torch.squeeze(x)


class SentenceRepresentation(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(SentenceRepresentation, self).__init__()
		self.input_size = input_size
		self.hidden = hidden_size
		self.num_layers = num_layers
		#could also consider adding dropout to 
		self.firstLayer = torch.nn.LSTM(input_size=input_size, 
										hidden_size=hidden_size, 
										num_layers=num_layers, 
										bidirectional=True)

	def forward(self, input_x):
		self.firstLayer = self.firstLayer.double()
		output = self.firstLayer(input_x.double())
		
		# padded_output, lengths = pad_packed_sequence(output)
		# batch_size = 5
		# maxes = Variable((torch.zeros(batch_size, padded_output.size(2))))
		# for i in range(batch_size):
		# 	maxes[i, :] = torch.max(output[:lengths[i], i, :], 0)[0]
		
		# return maxes
		# return torch.max(output)
		return output

	# def max_pooling(self, output, batch_size):
	# 	'''This function was adapted from https://github.com/koomri/text-segmentation/blob/874d6ef3ca0e402709b70924608d0894be9a93e1/models/max_sentence_embedding.py
	# 	'''
	# 	maxes = Variable((torch.zeros(batch_size, output.size(2))))
	# 	for i in range(batch_size):
	# 		maxes[i, :] = torch.max(output[:lengths[i], i, :], 0)[0]
		
	# 	return maxes
