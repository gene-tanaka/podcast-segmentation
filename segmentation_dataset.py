from pathlib2 import Path
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, word2Vec):
        self.word2Vec = word2Vec
        PAD_STR = '<pad>'
        BOUNDARY_STR = '<boundary>'
        self.examples = []
        self.targets = []
        all_objects = Path(root_dir).glob('**/*')
        self.filenames = [str(p) for p in all_objects if p.is_file() and str(p).split("/")[-1] != '.DS_Store']
        passages = []
        for filename in self.filenames:
            file = open(str(filename), "rt", encoding="utf8")
            raw_content = file.read()
            file.close()
            clean_txt = raw_content.strip()
            sentences = [s for s in clean_txt.split("\n") if len(s) > 0 and s != "\n"]
            passages.append(sentences)
        max_len = max([len(s) for s in passages])
        for passage in passages:
            target = [0]
            for i in range(1,len(passage)):
                if passage[i - 1][:3] == '===':
                    target[-1] = 1
                    passage[i - 1] = BOUNDARY_STR
                else:
                    target.append(0)
            target += [0]*(max_len - len(target))
            self.targets.append(target)
            example = [word for word in passage if word != BOUNDARY_STR]
            example += [PAD_STR]*(max_len - len(example))
            self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ret_vals = {}
        sentences = []
        ret_vals['target'] = self.targets[idx]
        for sentence in self.examples[idx]:
            sentence = []
            for word in sentence:
                if word in self.word2Vec:
                    sentence.append(self.word2Vec[word].reshape(1, 300))
                else:
                    sentence.append(self.word2Vec['UNK'].reshape(1, 300))
            sentences.append(sentence)
        ret_vals['sentences'] = sentences
        return ret_vals