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
def pk(ref: np.array, hyp: np.array, k: int = 5, boundary: int = 1):
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
    """

    if k is None:
        k = int(round(ref.shape[0] / (np.count_nonzero(ref[i : i + k] == boundary) * 2.0)))

    err = 0.0
    for i in range(len(ref) - k + 1):
        r = np.count_nonzero(ref[i : i + k] == boundary) > 0
        h = np.count_nonzero(hyp[i : i + k] == boundary) > 0
        if r != h:
            err += 1
    return err / (ref.shape[0] - k + 1.0)

'''
Adapted from nltk.metrics.segmentation https://www.nltk.org/_modules/nltk/metrics/segmentation.html
'''
def windowdiff(seg1: np.array, seg2: np.array, k: int = 5, boundary: int = 1, weighted: bool = False):
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
    """

    if seg1.shape[0] != seg2.shape[0]:
        raise ValueError("Segmentations have unequal length")
    if k > seg1.shape[0]:
        raise ValueError(
            "Window width k should be smaller or equal than segmentation lengths"
        )
    wd = 0.0
    for i in range(seg1.shape[0] - k + 1):
        ndiff = abs(np.count_nonzero(seg1[i : i + k] == boundary) - np.count_nonzero(seg2[i : i + k] == boundary))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (seg1.shape[0] - k + 1.0)

def validate(model, dataset):
    model.eval()
    total_pk = 0.0
    total_windowdiff = 0.0
    with tqdm(desc='Validating', total=len(dataset)) as pbar:
        for data in dataset:
            pbar.update()
            target = data['target']
            target = target.long()
            output = model(data['sentences'])
            output_softmax = F.softmax(output, 1)
            output_argmax = torch.argmax(output_softmax, dim=1)
            total_pk += pk(target.detach().numpy(), output_argmax.detach().numpy(), 5)
            total_windowdiff += windowdiff(target.detach().numpy(), output_softmax.detach().numpy(), 5)
    return total_pk, total_windowdiff

def train(model, num_epochs, train_set, dev_set, optimizer):
    model.train()
    total_loss = 0.0
    best_loss = float('inf')
    val_freq = 5
    model_save_path = 'saved_model'
    for i in range(num_epochs):
        print("Epoch {}:".format(i + 1))
        with tqdm(desc='Training', total=len(train_set)) as pbar:
            for data in train_set:
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
                if (i + 1) % val_freq == 0:
                    pk, windowdiff = validate(model, dev_set)
                    print("Pk: {}, WindowDiff: {}", pk, windowdiff)

        total_loss = total_loss / len(train_set)
        print("Total loss: " + str(total_loss))
        if total_loss < best_loss:
            best_loss = total_loss
            print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
            # model.save(model_save_path)
            torch.save(model, model_save_path)
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

    train_path = 'train_data'
    train_dataset = SegmentationDataset(train_path, word2vecModel)
    # train_dl = DataLoader(train_dataset)

    dev_path = 'val_data'
    dev_dataset = SegmentationDataset(dev_path, word2vecModel)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, 10, train_dataset, dev_dataset, optimizer)

if __name__ == '__main__':
    main()