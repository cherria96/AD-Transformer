import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

from ray import tune
from ray import train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
import wandb

#wandb
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting Model')

    # random seed
    parser.add_argument('--random_seed', type=int, default=100, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [Transformer, LSTM, DALSTM, RNN_LSTM, PatchFormer ]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='sbk_ad', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='sbk_ad.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--train_size', type=float, default=0.6, help='train size portion')
    parser.add_argument('--random_sample', type=bool, default=True, help='True: random samplingm, False: chronological sampling')
    parser.add_argument('--data_size', type=float, default=1.0, help='portion of data used for training, 1.0 means using all data')

    parser.add_argument('--target', type=str, default='BP', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--n_subs', type=int, default=16, help='number of substrate variables')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=30, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=15, help='start token length')
    parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')
    parser.add_argument('--series_decomposition', action='store_true', default=False, help='Series decomposition for encoder input')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size for moving average')
    parser.add_argument('--decoder_mode', type=str, default='default', help='decoder input mode, options: [default, past_subs, future_subs]')
    
    # PatchFormer
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    
    # Models 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=31, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=31, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=31, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=3, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--revin', action='store_true', default=False, help='RevIN')
    parser.add_argument('--wodenorm', action='store_true', default=False, help='w/o denormalization in RevIN')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    
    # hyperparameters
    parser.add_argument('--extra_tag', type=str, default="BasicVariantGenerator", help="Anything extra")
    args = parser.parse_args()
    
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    print('Args in experiment:')
    print(args)
    def getDict(args):
        return args.__dict__

    ###search space
    param_space = getDict(args)
    param_space_update = {
        # 'seq_len': tune.choice([64, 128, 256]),
        # 'label_len': tune.choice([16, 32, 64]),
        # 'pred_len': tune.choice([24, 48, 96]),

        ## model define
        'd_ff': tune.choice([16,32,64,128,256,512,1024,2048,4096]),
        'd_layers': tune.choice([1, 2, 3]),
        'd_model': tune.choice([16,32,64,128,256,512,1024,2048,4096]),
        'e_layers': tune.choice([1, 2, 3]),
        'factor': tune.choice([1, 2, 3, 4]),
        # 'd_model': tune.grid_search(list(args.d_model)),
        # 'd_model': args.d_model,
        'n_heads': tune.choice([2, 4, 8, 16]),

        ## optimization
        'batch_size': tune.choice([4, 16, 32, 64, 128, 256]),
        # 'itr': tune.choice([1, 2, 3, 4]),
        'learning_rate': tune.choice([0.00001, 0.0001, 0.001]),
        #'num_workers': tune.choice([8, 9, 10, 11, 12]),
        #'patience': tune.choice([1, 2, 3, 5, 7, 9, 11]),
        'train_epochs': tune.choice([1,2,3,4,5,6,7,8,9,10,11]),
        # 'e_layers': tune.choice([2, 4, 6]),
        # 'd_layers': tune.choice([1, 2, 3]),
        # 'd_ff': tune.choice(args.d_ff),
        # 'factor': tune.choice([3, 5, 7]),
        # 'embed': tune.choice(['fixed', 'learnable']),
        # 'distil': tune.choice([True, False])
    }

    ### Search algo
    algo = OptunaSearch()
    if param_space['extra_tag']=='OptunaSearch':
        algo = OptunaSearch()
    elif param_space['extra_tag']=='BayesOptSearch':
        algo = BayesOptSearch()
    elif param_space['extra_tag']=='AxSearch':
        algo = AxSearch()
    elif param_space['extra_tag']=='BasicVariantGenerator':
        #algo = BasicVariantGenerator()
        ##update ssearch space
        param_space_update = {
            ## model define
            'd_ff': tune.grid_search([16,32,64,128,256,512,1024,2048,4096]),
            'd_layers': tune.grid_search([1, 2, 3]),
            'd_model': tune.grid_search([16,32,64,128,256,512,1024,2048,4096]),
            'e_layers': tune.grid_search([1, 2, 3]),
            'factor': tune.grid_search([1, 2, 3, 4]),
            'n_heads': tune.grid_search([2, 4, 8, 16]),
            ## optimization
            'batch_size': tune.grid_search([4, 16, 32, 64, 128, 256]),
            # 'itr': tune.choice([1, 2, 3, 4]),
            'learning_rate': tune.grid_search([0.00001, 0.0001, 0.001]),
            'train_epochs': tune.grid_search([1,2,3,4,5,6,7,8,9,10,11]),
        }
    ### config
    if param_space['extra_tag']=='BasicVariantGenerator':
        tune_config=tune.TuneConfig(
            metric="vali_loss",
            mode="min",
            num_samples=20,
        )
    else:
        tune_config=tune.TuneConfig(
            metric="vali_loss",
            mode="min",
            search_alg=algo,
            num_samples=20,
        )
    param_space.update(param_space_update)

    
    def train_fn(param_space):
        # wandb = setup_wandb(param_space)
        print('is cuda avaiable ', torch.cuda.is_available())
        
        Exp = Exp_Main
        

        if param_space['is_training']:
            for ii in range(param_space['itr']):
                # setting record of experiments
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    param_space['model_id'],
                    param_space['model'],
                    param_space['data'],
                    param_space['features'],
                    param_space['seq_len'],
                    param_space['label_len'],
                    param_space['pred_len'],
                    param_space['d_model'],
                    param_space['n_heads'],
                    param_space['e_layers'],
                    param_space['d_layers'],
                    param_space['d_ff'],
                    param_space['factor'],
                    param_space['embed'],
                    param_space['distil'],
                    param_space['des'],ii)

                exp = Exp(param_space)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                if param_space['do_predict']:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(param_space['model_id'],
                                                                                                        param_space['model'],
                                                                                                        param_space['data'],
                                                                                                        param_space['features'],
                                                                                                        param_space['seq_len'],
                                                                                                        param_space['label_len'],
                                                                                                        param_space['pred_len'],
                                                                                                        param_space['d_model'],
                                                                                                        param_space['n_heads'],
                                                                                                        param_space['e_layers'],
                                                                                                        param_space['d_layers'],
                                                                                                        param_space['d_ff'],
                                                                                                        param_space['factor'],
                                                                                                        param_space['embed'],
                                                                                                        param_space['distil'],
                                                                                                        param_space['des'], ii)

            exp = Exp(param_space)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
            
    trainable = tune.with_resources(
        tune.with_parameters(train_fn),
        {"gpu": 1},
    )
    #The BasicVariantGenerator is used per default if no search algorithm is passed to Tuner.
    tuner = tune.Tuner(
        trainable,
        tune_config=tune_config,
        param_space=param_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    wandb.finish()