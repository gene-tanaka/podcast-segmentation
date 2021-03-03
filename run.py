from segmentation_dataset import SegmentationDataset
from model import Model
import io
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sys
import numpy as np

'''
Adapted from nltk.metrics.segmentation https://www.nltk.org/_modules/nltk/metrics/segmentation.html
'''
def pk(ref, hyp, k=None, boundary="1"):
    """
    Compute the Pk metric for a pair of segmentations A segmentation
    is any sequence over a vocabulary of two items (e.g. "0", "1"),
    where the specified boundary value is used to mark the edge of a
    segmentation.

    >>> '%.2f' % pk('0100'*100, '1'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0100'*100, 2)
    '0.00'

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the segmentation to evaluate
    :type hyp: str or list
    :param k: window size, if None, set to half of the average reference segment length
    :type boundary: str or int or bool
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """

    if k is None:
        k = int(round(len(ref) / (ref.count(boundary) * 2.0)))

    err = 0
    for i in range(len(ref) - k + 1):
        r = ref[i : i + k].count(boundary) > 0
        h = hyp[i : i + k].count(boundary) > 0
        if r != h:
            err += 1
    return err / (len(ref) - k + 1.0)

'''
Adapted from nltk.metrics.segmentation https://www.nltk.org/_modules/nltk/metrics/segmentation.html
'''
def windowdiff(seg1, seg2, k, boundary="1", weighted=False):
    """
    Compute the windowdiff score for a pair of segmentations.  A
    segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

        >>> s1 = "000100000010"
        >>> s2 = "000010000100"
        >>> s3 = "100000010000"
        >>> '%.2f' % windowdiff(s1, s1, 3)
        '0.00'
        >>> '%.2f' % windowdiff(s1, s2, 3)
        '0.30'
        >>> '%.2f' % windowdiff(s2, s3, 3)
        '0.80'

    :param seg1: a segmentation
    :type seg1: str or list
    :param seg2: a segmentation
    :type seg2: str or list
    :param k: window width
    :type k: int
    :param boundary: boundary value
    :type boundary: str or int or bool
    :param weighted: use the weighted variant of windowdiff
    :type weighted: boolean
    :rtype: float
    """

    if len(seg1) != len(seg2):
        raise ValueError("Segmentations have unequal length")
    if k > len(seg1):
        raise ValueError(
            "Window width k should be smaller or equal than segmentation lengths"
        )
    wd = 0
    for i in range(len(seg1) - k + 1):
        ndiff = abs(seg1[i : i + k].count(boundary) - seg2[i : i + k].count(boundary))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (len(seg1) - k + 1.0)

def train(model, num_epochs, dataset, optimizer):
	model.train()
	total_loss = 0.0
	epoch_num = 0
	while epoch_num < num_epochs:
		with tqdm(desc='Training', total=len(dataset)) as pbar:
			for data in dataset:
				pbar.update()
				model.zero_grad()
				output = model(data['sentences'])
				target = data['target']
				target = target.long()
				loss = model.loss(output, target)
				loss.backward()
				optimizer.step()
				total_loss += loss
				pbar.set_description('Training, loss={:.4}'.format(loss))
	total_loss = total_loss / len(dataset)
	print("Total loss: " + str(total_loss))
	model_save_path = 'saved_model'
	print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
	model.save(model_save_path)
	# also save the optimizers' state
	torch.save(optimizer.state_dict(), model_save_path + '.optim')


'''
The following function was taken from https://fasttext.cc/docs/en/english-vectors.html
'''
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:]).astype(np.float)
    return data

def main():
    word2vecModel = load_vectors('wiki-news-300d-1M-subword.vec')
    # word2vecModel = {"UNK": np.zeros((1,300))} # dummy data

    train_dataset = SegmentationDataset('data', word2vecModel)
    # train_dl = DataLoader(train_dataset)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, 500, train_dataset, optimizer)

main()