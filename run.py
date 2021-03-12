from segmentation_dataset import SegmentationDataset
from model import Model
from baseline import Baseline
from nltk import pk, windowdiff
import io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sys
import mmap
import numpy as np

def validate(model, dataset, indices):
    model.eval()
    total_pk = 0.0
    total_windowdiff = 0.0
    with tqdm(desc='Validating', total=5) as pbar:
        for i, data in enumerate(dataset):
            if i in indices:
                pbar.update()
                target = torch.flatten(data['target'], start_dim=0, end_dim=1)
                target = target.long()
                output = model(torch.flatten(data['sentences'], start_dim=0, end_dim=1))
                output_softmax = F.softmax(output, 1)
                target_list = target.tolist()
                output_list = torch.argmax(output_softmax, dim=1).tolist()
                k = int(round(len(target_list) / (target_list.count(1) * 2.0)))
                total_pk += pk(target_list, output_list, k=k, boundary=1)
                total_windowdiff += windowdiff(target_list, output_list, k=k, boundary=1)
    return total_pk / len(indices), total_windowdiff / len(indices)

def train(model, num_epochs, train_set, dev_set, optimizer):
    print()
    print("Starting training...")
    model.train()
    total_loss = 0.0
    best_loss = float('inf')
    model_save_path = 'saved_model'
    for i in range(num_epochs):
        print("Epoch {}/{}:".format(i + 1, num_epochs))
        with tqdm(desc='Training', total=5) as pbar:
            indices = [np.random.randint(0, len(train_set)) for i in range(5)]
            for i, data in enumerate(train_set):
                if i not in indices:
                    # print("not doing this one ")
                    continue
                else:
                    # data = train_set[i]
                    pbar.update()
                    model.zero_grad()
                    # output = model(data['sentences'][i_batch, :, :, :])
                    output = model(torch.flatten(data['sentences'], start_dim=0, end_dim=1))
                    # target = data['target'][i_batch]
                    target = torch.flatten(data['target'], start_dim=0, end_dim=1)
                    target = target.long()
                    loss = model.loss(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss
                    pbar.set_description('Training, loss={:.4}'.format(loss))
        total_loss = total_loss / 5
        print("Total loss: {}".format(total_loss))
        if total_loss < best_loss:
            best_loss = total_loss
            print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
            torch.save(model, model_save_path)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), model_save_path + '.optim')
    return indices
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
    # word2vecModel = load_vectors('wiki-news-300d-1M-subword.vec')
    word2vecModel = {"UNK": np.zeros((1,300))} # dummy data

    train_path = 'wiki_50'
    train_dataset = SegmentationDataset(train_path, word2vecModel)
    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True)

    dev_path = 'wiki_50'
    dev_dataset = SegmentationDataset(dev_path, word2vecModel)
    dev_dl = DataLoader(dev_dataset, batch_size=4, shuffle=True)

    model = Model(attention=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    indices = train(model, 5, train_dl, dev_dl, optimizer)
    pk, windowdiff = validate(model, dev_dl, indices)
    print("Pk: {}, WindowDiff: {}".format(pk, windowdiff))

    baseline_threshold = 5.0
    baseline = Baseline(dev_dl, baseline_threshold)
    base_pk, base_windowdiff = baseline.evaluate()
    print("Baseline Pk: {}, Baseline Window Diff: {}".format(base_pk, base_windowdiff))

if __name__ == '__main__':
    main()