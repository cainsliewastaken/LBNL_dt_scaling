import torch
from torch.utils.data import Dataset
from load_data import *

class Multistep_TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, lead):
        self.data = data # Your complete time-series data (e.g., a NumPy array or list)
        self.sequence_length = sequence_length
        self.lead = lead

    def __len__(self):
        # Number of possible starting points for sequences
        # print(len(self.data), self.sequence_length*self.lead)
        return len(self.data) - self.sequence_length*self.lead + 1

    def __getitem__(self, idx):
        # Extract a sequence of 'sequence_length' starting from 'idx'
        input_sequence = self.data[idx : idx + self.lead*self.sequence_length:self.lead]
        # Optionally, if you have a target for the sequence
        # target = self.data[idx + self.sequence_length] 
        return input_sequence
    
class Multistep_TimeSeriesDataset_load_from_file(Dataset):
    def __init__(self, start_ind, total_len, sequence_length, lead, m1, s1):
        # self.data = data # Your complete time-series data (e.g., a NumPy array or list)
        self.sequence_length = sequence_length
        self.lead = lead
        self.loading_func = load_data_with_lead #is is a function loaded in from load_data.py at the top of the file
        self.total_len = total_len
        self.m1 = m1
        self.s1 = s1
        self.start_ind = start_ind

    def __len__(self):
        # Number of possible starting points for sequences
        # print(self.total_len, self.sequence_length*self.lead)
        return self.total_len - self.sequence_length*self.lead + 1

    def __getitem__(self, idx):
        # Extract a sequence of 'sequence_length' starting from 'idx'
        return (self.loading_func(self.start_ind + idx, self.sequence_length, self.lead) - self.m1)/self.s1
    