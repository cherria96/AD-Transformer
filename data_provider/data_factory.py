from data_provider.data_loader_orig import TimeSeriesDataset
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
        train_size = args.train_size,
        seq_len = args.seq_len,
        label_len = args.label_len,
        pred_len = args.pred_len,
        scale = True,
        inverse = inverse,
        random_sample=args.random_sample,
        data_size = args.data_size,
        cv_splits = args.cv_splits,
        cv_index = args.cv_index,
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

    
