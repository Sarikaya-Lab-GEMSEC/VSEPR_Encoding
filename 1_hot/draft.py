from data import make_matrix
import pandas as pd
import numpy as np
import os
import math
import time
import torch
import model as net
from Criteria import MSELoss
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import DataSet as myDataLoader
from sklearn.model_selection import KFold
import VisualizeGraph as viz
from torch.autograd import Variable
from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr



def cross_val(args):

    torch.set_default_tensor_type('torch.DoubleTensor')

    csv_path = os.path.join(args.data_dir, args.file_path)
    data_ori = pd.read_csv(csv_path)
    data_ori = data_ori.loc[data_ori['species'] == 'human']
    data_ori = data_ori.loc[(data_ori['peptide_length'] == 9) | (data_ori['peptide_length'] == 10)]
    #allele = data['mhc'].drop_duplicates()

    allele_list_9 = ['HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*02:03','HLA-B*15:01','HLA-A*31:01','HLA-A*01:01','HLA-B*07:02','HLA-A*26:01',
              'HLA-A*02:06','HLA-A*68:02','HLA-B*08:01','HLA-B*58:01','HLA-B*40:01','HLA-B*27:05','HLA-A*30:01','HLA-A*69:01','HLA-B*57:01','HLA-B*35:01',
              'HLA-A*02:02','HLA-A*24:02','HLA-B*18:01','HLA-B*51:01','HLA-A*29:02','HLA-A*68:01','HLA-A*33:01','HLA-A*23:01']

    allele_list_10 = ['HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*68:01','HLA-A*31:01','HLA-A*02:06','HLA-A*68:02','HLA-A*02:03','HLA-A*33:01','HLA-A*02:02']



    for length in [9, 10]:

        if length == 9:
            allele_list = allele_list_9
        elif length == 10:
            allele_list = allele_list_10
        else:
            print("Invalid Length")
            exit(0)

        for allele in allele_list:   #[9,10]

            model_dir = args.savedir + os.sep + 'best_model' + os.sep + allele
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            data = data_ori.loc[(data_ori['peptide_length'] == length)]
            data = data.loc[data_ori['mhc'] == allele]
            sequ = data['sequence'].values.tolist()
            matrix_list = [make_matrix(x) for x in sequ]
            meas = data['meas'].values.tolist()
            bind = []
            positive = [i for i in meas if i< 500]
            print(len(positive), len(meas) - len(positive))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_path', default="iedb.csv")
    parser.add_argument('--model', default="shallow_net")  #
    parser.add_argument('--data_dir', default="./data/")  # data directory
    parser.add_argument('--max_epochs', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--step_loss', type=int, default=20)
    parser.add_argument('--lr', type=float, default= 0.01)
    parser.add_argument('--savedir', default='./results_shallow')
    parser.add_argument('--crossValFile', default='crossValFile.txt')
    parser.add_argument('--visualizeNet', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--onGPU', default=True)


    cross_val(parser.parse_args())










