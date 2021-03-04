import numpy as np
import torch

class Baseline():
	def __init__(self, data, threshold):
		self.dataset = data
		self.threshold = threshold
		
	def getBaselineLabels(self):
		labels = [0 for i in range(self.dataset[0])]
		for data in self.dataset:
			for i in range(data.shape[0] -1):
				firstSentenceTensor = data[i, :, :]
				secondSentenceTensor = data[i+1, :, :]
				if np.sum(abs(secondSentenceTensor - firstSentenceTensor)) > self.threshold:
					labels[i] = 1
		return torch.Tensor(labels)

