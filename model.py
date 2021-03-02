import torch

class Model(self):
	def __init__(self, init_1, hideen_1, init_2,  hidden_2, init_3, hidden_3):
		self.SentenceRepresentation = SentenceRepresentation(init1, hidden1)
		self.secondNN = SecondNN(init2, hidden2)
		self.lastLayer = LastLayer(init3, hidden3)

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