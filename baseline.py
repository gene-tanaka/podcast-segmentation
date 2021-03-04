from metrics import pk, windowdiff
import numpy as np
import torch
from tqdm import tqdm

class Baseline():
    def __init__(self, data, threshold):
        self.dataset = data
        self.data_len = len(data)
        self.threshold = threshold
        self.all_labels = []
        self.getBaselineLabels()
        
    def getBaselineLabels(self):
        for data in self.dataset:
            sentence_labels = [0 for i in range(data['target'].shape[0])]
            sentences = data['sentences']
            for i in range(1, sentences.shape[0]):
                firstSentenceTensor = sentences[i - 1, :, :]
                secondSentenceTensor = sentences[i, :, :]
                if torch.norm(secondSentenceTensor - firstSentenceTensor).item() > self.threshold:
                    sentence_labels[i - 1] = 1
            self.all_labels.append(sentence_labels)
    
    def evaluate(self):
        total_pk = 0.0
        total_windowdiff = 0.0
        with tqdm(desc='Validating Baseline', total=self.data_len) as pbar:
            for i in range(self.data_len):
                data = self.dataset[i]
                pbar.update()
                target = data['target']
                target = target.long()
                pred = self.all_labels[i]
                total_pk += pk(target.detach().numpy(), np.array(pred))
                total_windowdiff += windowdiff(target.detach().numpy(), np.array(pred))
        return total_pk / self.data_len, total_windowdiff / self.data_len