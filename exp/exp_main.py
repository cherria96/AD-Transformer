from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, LSTM, DALSTM, RNN_LSTM, PatchFormer
# from kan import KAN
from utils.tools import EarlyStopping, adjust_learning_rate, visualize_predictions, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'LSTM': LSTM,
            'DALSTM': DALSTM,
            'RNN_LSTM': RNN_LSTM,
            'Patchformer': PatchFormer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def cluster_loss(self, M, S):
        """
        Compute the cluster loss L_C.
        
        S: [C, C] - Channel similarity matrix
        M: [C, K] - Cluster assignment matrix (C channels, K clusters)
        
        Returns:
        L_C: Cluster loss
        """
        # Compute M^T S M
        within_cluster_similarity = torch.trace(torch.mean(M.transpose(1,2) @ S @ M, dim = 0))
        
        # Compute (I - M M^T) S
        I = torch.eye(M.shape[1], device=M.device)  # Identity matrix of size [C, C]
        between_cluster_separation = torch.trace(torch.mean((I - M @ M.transpose(1,2)) @ S, dim = 0))
        
        # Cluster loss: L_C = -Tr(M^T S M) + Tr((I - M M^T) S)
        L_C = -within_cluster_similarity + between_cluster_separation
        
        return L_C
    def _decoder_input(self, batch_x, batch_y):
        batch_x = batch_x.cpu()  # (B, S, D)
        batch_y = batch_y.cpu()  # (B, T, D)
        if self.args.decoder_mode == 'default':
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        # elif self.args.decoder_mode == 'past_subs':
        #     dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.n_subs:]).float()
            
        #     full_seq = torch.cat([batch_x[:, :, :self.args.n_subs], batch_y[:, :, :self.args.n_subs]], dim = 1).float()
        #     dec_inp = torch.cat()
            
        #     dec_inp = torch.cat([batch_x[:, -self.args.pred_len:, :self.args.n_subs], dec_inp], dim=-1).float()
        #     dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        elif self.args.decoder_mode == 'future_subs':
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.n_subs:]).float()
            dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.n_subs], dec_inp], dim=-1).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        
        return dec_inp

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                # batch_y = batch_y[:,:,4:]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._decoder_input(batch_x, batch_y)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DNN' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif  'LSTM' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)
                        elif 'CCM' in self.args.model:
                            outputs, M, S = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DNN' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif  'LSTM' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)

                    elif 'CCM' in self.args.model:
                        outputs, M, S = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.model == 'CARD':
                        self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs * self.ratio
                        batch_y = batch_y * self.ratio
                # if f_dim!=-1:
                #         outputs = outputs[:,:,4:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                if 'CCM' in self.args.model:
                    loss += self.args.beta * self.cluster_loss(M, S).detach().cpu()
                    

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        total_start_time = time.time()  
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        c = nn.L1Loss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
        #                                     steps_per_epoch = train_steps,
        #                                     pct_start = self.args.pct_start,
        #                                     epochs = self.args.train_epochs,
        #                                     max_lr = self.args.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(model_optim, gamma = 0.9)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._decoder_input(batch_x, batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DNN' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif  'LSTM' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)

                        elif 'CCM' in self.args.model:
                            outputs, M, S = self.model(batch_x)

                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'DNN' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    elif  'LSTM' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)

                    elif 'CCM' in self.args.model:
                        outputs, M, S = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.model == 'CARD':
                        self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs * self.ratio
                        batch_y = batch_y * self.ratio

                    # if f_dim!=-1:
                    #     outputs = outputs[:,:,:]
                    
                    
                    if 'CARD' in self.args.model:
                        loss = c(outputs, batch_y)  
                    else:
                        loss = criterion(outputs, batch_y)
                    if 'CCM' in self.args.model:
                        loss += self.args.beta * self.cluster_loss(M, S)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        total_end_time = time.time()
        print(f"Total Training Time: {total_end_time - total_start_time:.2f} seconds")
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        target_mean = test_data.scaler.mean_[-1]
        target_std = np.sqrt(test_data.scaler.var_)[-1]

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._decoder_input(batch_x, batch_y)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DNN' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif  'LSTM' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)

                        elif 'CCM' in self.args.model:
                            outputs, M, S = self.model(batch_x)

                        else:
                            if self.args.output_attention:
                                outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DNN' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    elif  'LSTM' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)

                    elif 'CCM' in self.args.model:
                        outputs, M, S = self.model(batch_x)

                    else:
                        if self.args.output_attention:
                            outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # if f_dim !=-1:
                #     outputs = outputs[:,:,4:]
                
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        if self.args.output_attention:
            attns = np.array(torch.cat(attn).cpu())

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        visualize_predictions(preds.squeeze(), trues.squeeze(), seq_len = self.args.seq_len, pred_len = self.args.pred_len, col = 'MY', 
                                name = os.path.join(folder_path, 'plot.pdf'))

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse: {}, rse:{}'.format(mse, mae, rmse, rse))
        f = open("results_ablation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse: {}, rse:{}'.format(mse, mae, rmse, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        print('prediction shape:',preds.shape)
        print('groundtruth shape:',trues.shape)
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        if self.args.output_attention:
            np.save(folder_path + 'attn.npy', attns)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = self._decoder_input(batch_x, batch_y)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DNN' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif  'LSTM' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark)

                        elif 'CCM' in self.args.model:
                            outputs, M, S = self.model(batch_x)

                        else:
                            if self.args.output_attention:
                                outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            elif 'CCM' in self.args.model:
                                outputs, M, S = self.model(batch_x)

                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DNN' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif  'LSTM' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark)
                    elif 'CCM' in self.args.model:
                        outputs, M, S = self.model(batch_x)

                    else:
                        if self.args.output_attention:
                            outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return