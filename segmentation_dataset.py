from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from pathlib2 import Path

class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, word2Vec):
        self.word2Vec = word2Vec
        PAD_STR = '<pad>'
        BOUNDARY_STR = '<boundary>'
        self.examples = []
        self.targets = []
        all_objects = Path(root_dir).glob('**/*')
        self.filenames = [str(p) for p in all_objects if p.is_file() and str(p).split("/")[-1] != '.DS_Store']
        # print(self.filenames)
        passages = []
        print()
        print("Reading raw data...")
        with tqdm(desc='Progress', total=len(self.filenames)) as pbar:
            for filename in self.filenames:
                pbar.update()
                file = open(str(filename), "rt", encoding="utf8")
                raw_content = file.read()
                file.close()
                clean_txt = raw_content.strip()
                sentences = [s for s in clean_txt.split("\n") if len(s) > 0 and s != "\n"]
                passages.append(sentences)
        self.max_len_passage = max([len(s) for s in passages])
        self.max_len_sentence = 0
        print()
        print("Converting documents to embeddings...")
        with tqdm(desc='Progress', total=len(self.filenames)) as pbar:
            for passage in passages:
                pbar.update()
                target = [0]
                for i in range(1,len(passage)):
                    if passage[i - 1][:3] == '===':
                        target[-1] = 1
                        passage[i - 1] = BOUNDARY_STR
                    else:
                        target.append(0)
                target += [0]*(self.max_len_passage - len(target))
                self.targets.append(target)
                example = [sentence for sentence in passage if sentence != BOUNDARY_STR]
                for sentence in passage:
                    self.max_len_sentence = max(self.max_len_sentence, len(sentence.split(' ')))
                example += [PAD_STR]*(self.max_len_passage - len(example))
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ret_vals = {}
        sentence_tensors = []
        ret_vals['target'] = torch.Tensor(self.targets[idx])
        for sentence in self.examples[idx]:
            sentence_embedding = []
            for word in sentence.split(' '):
                if word in self.word2Vec:
                    sentence_embedding.append(torch.from_numpy(self.word2Vec[word].reshape(1, 300)))
                else:
                    sentence_embedding.append(torch.from_numpy(self.word2Vec['UNK'].reshape(1, 300)))
            sentence_embedding += [torch.zeros(1,300)]*(self.max_len_sentence - len(sentence_embedding))
            sentence_tensors.append(torch.stack(sentence_embedding))

        ret_vals['sentences'] = torch.squeeze(torch.stack(sentence_tensors, dim=0))
        return ret_vals