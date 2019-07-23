import torch
import numpy as np
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pepList, labelList):
        pepList = [torch.from_numpy(np.array(x)) for x in pepList]
        labelList = [torch.DoubleTensor([x]) for x in labelList]
        self.pepList = pepList
        self.labelList = labelList

    def __len__(self):
        return len(self.pepList)

    def __getitem__(self, idx):
        pep_sequ = self.pepList[idx]
        label = self.labelList[idx]
        return (pep_sequ, label)
