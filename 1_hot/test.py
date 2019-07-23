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
from sklearn.metrics import roc_auc_score
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import pickle


def val(args, val_loader, model, criterion):
#    args.onGPU=False
    with torch.no_grad():

        output_list = []
        label_list = []

        epoch_loss = []
        epoch_MSE = []

        total_batches = len(val_loader)
        for i, (input, target) in enumerate(val_loader):
            start_time = time.time()

            if args.onGPU == True:
                input = input.cuda()
                target = target.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # run the mdoel
            output = model(input_var)

            output_list.extend(output.detach().data.cpu().numpy().flatten().tolist())
            label_list.extend(target.cpu().numpy().flatten().tolist())

            # compute the loss
            loss = criterion(output.view(1, len(input))[0], target_var)

            epoch_loss.append(loss.item())

            epoch_MSE.append(mean_squared_error(target.cpu().numpy(), output.detach().data.cpu().numpy()))

            time_taken = time.time() - start_time

            print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

        average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
        average_epoch_MSE = sum(epoch_MSE) / len(epoch_MSE)
    

    return average_epoch_loss_val, average_epoch_MSE, output_list, label_list


def cross_val(args):

    torch.set_default_tensor_type('torch.DoubleTensor')

    csv_path = os.path.join(args.data_dir, args.file_path)
    data_ori = pd.read_csv(csv_path)
    data_ori = data_ori.loc[data_ori['species'] == 'human']
    data_ori = data_ori.loc[(data_ori['peptide_length'] == 9) | (data_ori['peptide_length'] == 10)]

    allele_list_9 = ['HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*02:03','HLA-B*15:01','HLA-A*31:01','HLA-A*01:01','HLA-B*07:02','HLA-A*26:01','HLA-A*02:06','HLA-A*68:02','HLA-B*08:01','HLA-B*58:01','HLA-B*40:01','HLA-B*27:05','HLA-A*30:01','HLA-A*69:01','HLA-B*57:01','HLA-B*35:01', 'HLA-A*02:02','HLA-A*24:02','HLA-B*18:01','HLA-B*51:01','HLA-A*29:02','HLA-A*68:01','HLA-A*33:01','HLA-A*23:01']

    allele_list_10 = ['HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*68:01','HLA-A*31:01','HLA-A*02:06','HLA-A*68:02','HLA-A*02:03','HLA-A*33:01','HLA-A*02:02']

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)


    logFileLoc = args.savedir + os.sep + args.testFile

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
        logger.write("%s\t%s\t\t\t%s\t\t\t%s\t\t\t%s\n" % ('Length', 'Allele', 'Pearson', 'AUC', 'SRCC'))
        logger.flush()
    else:
        logger = open(logFileLoc, 'w')
        logger.write("%s\t%s\t\t\t%s\t\t\t%s\t\t\t%s\n" % ('Length', 'Allele', 'Pearson', 'AUC', 'SRCC'))
        logger.flush()

    for length in [10, 9]:

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

            for i in meas:
                i = (-1) * math.log10(i);
                bind.append(i)
            sequ, label = matrix_list, bind

            if (len(sequ) > 0):

                sequ_ori, label_ori = sequ, label

                output_list=[]
                label_list=[]

                fold_num = 0

                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                for train_set, test_set in kf.split(sequ_ori, label_ori):

                    fold_num += 1

                    train_sequ, test_sequ, train_label, test_label = [sequ_ori[i] for i in train_set], [sequ_ori[i] for i in test_set],\
                                                                [label_ori[i] for i in train_set], [label_ori[i] for i in test_set]

                    test_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(test_sequ, test_label),
                                                                 batch_size=args.batch_size, shuffle=True,
                                                                 num_workers=args.num_workers, pin_memory=True)

                    model = net.GRU_net()

                    if args.onGPU == True:
                        #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
                        model = model.cuda()

                    criteria = MSELoss()

                    if args.onGPU == True:
                        criteria = criteria.cuda()

                    best_model_dict = torch.load(model_dir + os.sep + allele + '_' + str(length) + '_' + str(fold_num) + '.pth')
                    model.load_state_dict(best_model_dict)
                    _, _, output, label = val(args, test_data_load, model, criteria)

                    output_list.extend(output)
                    label_list.extend(label)
                
                #IC_output_list = [math.pow(10, (-1) * value) for value in output_list]
                #IC_label_list = [math.pow(10, (-1) * value) for value in label_list]

                IC_output_list = output_list
                IC_label_list = label_list

                bi_output_list = [1 if ic > (-1) * math.log10(500) else 0 for ic in IC_output_list]
                bi_label_list = [1 if ic > (-1) * math.log10(500) else 0 for ic in IC_label_list]

                pearson = pearsonr(IC_output_list, IC_label_list)
                auc = roc_auc_score(bi_label_list, bi_output_list)
                srcc = spearmanr(IC_label_list, IC_output_list)

                logger.write("%s\t%s\t\t%.4f\t\t\t%.4f\t\t\t%.4f\n" % (length, allele, pearson[0], auc, srcc[0]))
                logger.flush()

                prediction = args.savedir + os.sep + args.predict
                if os.path.exists(prediction):
                    append_write = 'a' # append if already exists
                else:
                    append_write = 'w' 

                true_value = open(prediction, append_write)
                true_value.write("%s\n" % (allele))
                for i in range(int(len(output_list) / 10)):
                    true_value.write("%.4f\t%.4f\n" % (IC_label_list[i], IC_output_list[i]))
                true_value.flush()
                    
    logger.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_path', default="iedb.csv")
    parser.add_argument('--data_dir', default="./data/")  # data di:rectory
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--step_loss', type=int, default=5)
    parser.add_argument('--lr', type=float, default= 0.01)
    parser.add_argument('--savedir', default='./results_gru')
    parser.add_argument('--testFile', default='testFile2.txt')
    parser.add_argument('--predict', default='prediction.txt')
    parser.add_argument('--visualizeNet', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--onGPU', default=True)


    cross_val(parser.parse_args())










