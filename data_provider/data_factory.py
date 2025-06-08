from data_provider.data_loader import TimeSeriesDataset, AnomalyDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd

def data_provider(args, flag):
    Data = TimeSeriesDataset
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        inverse = False 
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        inverse = False 
        

    data_set = Data(
        path = args.root_path + args.data_path,
        split = flag,
        seq_len = args.seq_len,
        label_len = args.label_len,
        pred_len = args.pred_len,
        scale = True,
        inverse = inverse
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

    
def get_loader_segment(args, flag):
    dataset = AnomalyDataset(args.root_path + args.data_path, args.win_size, args.step, flag)

    shuffle = False
    if flag == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader

    
