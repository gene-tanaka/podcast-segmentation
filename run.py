from segmentation_dataset import SegmentationDataset
from model import Model
import io
import torch

def train(model, num_epochs, dataset, optimizer):
	model.train()
	total_loss = 0.0
	epoch_num = 0
	while epoch_num < num_epochs:
		for data in dataset:
			model.zero_grad()
			output = model(data['sentences'])
			target = data['target']
			loss = model.criterion(output, target)
			loss.backward()
			optimizer.step()
			total_loss += loss.data[0]
	total_loss = total_loss / len(dataset)
	print("Total loss: " + str(total_loss))
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

train_dataset = SegmentationDataset('data', word2vecModel)

model = Model()
optimizer = torch.optim.Adam(model.parameters, lr=1e-4)
train(model, 500, train_dataset, optimizer)