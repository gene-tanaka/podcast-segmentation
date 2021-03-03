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
		self.secondNN = self.secondNN.double()
		secondNN_output, _ = self.secondNN(encoded_batch)
		self.lastLayer = self.lastLayer.double()
		x = self.lastLayer(secondNN_output)
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
		output_mean = torch.mean(output[0], dim=1) # also try torch.max
		output_mean = torch.unsqueeze(output_mean, dim=1)

		return (output_mean, output[1])
