from segmentation_dataset import SegmentationDataset
from model import Model
from baseline import Baseline
from metrics import pk, windowdiff
import io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sys
import mmap
import numpy as np

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
            total_pk += pk(target.detach().numpy(), output_argmax.detach().numpy())
            total_windowdiff += windowdiff(target.detach().numpy(), output_softmax.detach().numpy())
    return total_pk / len(dataset), total_windowdiff / len(dataset)

def train(model, num_epochs, train_set, dev_set, optimizer):
    print()
    print("Starting training...")
    model.train()
    total_loss = 0.0
    best_loss = float('inf')
    val_freq = 1
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
            print("Pk: {}, WindowDiff: {}".format(pk, windowdiff))

        total_loss = total_loss / len(train_set)
        print("Total loss: {}".format(total_loss))
        if total_loss < best_loss:
            best_loss = total_loss
            print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
            # model.save(model_save_path)
            torch.save(model, model_save_path)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), model_save_path + '.optim')

'''
Adapted from https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
'''
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

'''
The following function was adapted from https://fasttext.cc/docs/en/english-vectors.html
'''
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    print("Loading word2vec embeddings...")
    with tqdm(desc='Progress', total=get_num_lines(fname)) as pbar:
        for line in fin:
            pbar.update()
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:]).astype(np.float)
    return data

def main():
    word2vecModel = load_vectors('wiki-news-300d-1M-subword.vec')
    # word2vecModel = {"UNK": np.zeros((1,300))} # dummy data

    train_path = 'train_data'
    train_dataset = SegmentationDataset(train_path, word2vecModel)
    # train_dl = DataLoader(train_dataset)

    dev_path = 'wiki_50'
    dev_dataset = SegmentationDataset(dev_path, word2vecModel)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, 10, train_dataset, dev_dataset, optimizer)

    baseline_threshold = 5.0
    baseline = Baseline(dev_dataset, baseline_threshold)
    base_pk, base_windowdiff = baseline.evaluate()
    print("Baseline Pk: {}, Baseline Window Diff: {}".format(base_pk, base_windowdiff))

if __name__ == '__main__':
    main()