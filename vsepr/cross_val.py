from data import make_matrix
import pandas as pd
import numpy as np
import os
import math
import time
import torch
import model_vsepr as net
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




def train(args, train_loader, model, criterion, optimizer):
    # switch to train mode
#    args.onGPU = True

    model.train()

    epoch_loss = []
    epoch_MSE = []

    total_batches = len(train_loader)

    for i, (sequ, target) in enumerate(train_loader):
        start_time = time.time()

        if args.onGPU == True:
            sequ = sequ.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(sequ)
        target_var = torch.autograd.Variable(target)

        optimizer.zero_grad()

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output.view(1, len(sequ))[0], target_var)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        epoch_MSE.append(mean_squared_error(target.cpu().numpy(), output.cpu().detach().data.numpy()))

        time_taken = time.time() - start_time

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    average_epoch_MSE = sum(epoch_MSE) / len(epoch_MSE)

    return average_epoch_loss_val, average_epoch_MSE

def cross_val(args):

    torch.set_default_tensor_type('torch.DoubleTensor')

    allele_list_9 = ['HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*02:03','HLA-B*15:01','HLA-A*31:01','HLA-A*01:01','HLA-B*07:02','HLA-A*26:01','HLA-A*02:06','HLA-A*68:02','HLA-B*08:01','HLA-B*58:01','HLA-B*40:01','HLA-B*27:05','HLA-A*30:01','HLA-A*69:01','HLA-B*57:01','HLA-B*35:01', 'HLA-A*02:02','HLA-A*24:02','HLA-B*18:01','HLA-B*51:01','HLA-A*29:02','HLA-A*68:01','HLA-A*33:01','HLA-A*23:01']

    allele_list_10 = ['HLA-A*02:01','HLA-A*03:01','HLA-A*11:01','HLA-A*68:01','HLA-A*31:01','HLA-A*02:06','HLA-A*68:02','HLA-A*02:03','HLA-A*33:01','HLA-A*02:02']

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.visualizeNet == True:
        x = Variable(torch.randn(1, 5, 174, 18))
        model = net.ResNetC1()

        if args.onGPU == True:
            x = x.cuda()
            model = model.cuda()

        y = model.forward(x)
        #g = viz.make_dot(y)
        #g.render(args.savedir + '/model.png', view=False)

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p

        print('Parameters: ' + str(total_paramters))


    logFileLoc = args.savedir + os.sep + args.crossValFile

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
        logger.write("%s\t%s\t\t\t\t\t%s\n" % ('Length', 'Allele', 'Pearson'))
        logger.flush()
    else:
        logger = open(logFileLoc, 'w')
        logger.write("%s\t%s\t\t\t\t\t%s\n" % ('Length', 'Allele', 'Pearson'))
        logger.flush()

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

            data_dict = pickle.load(open(args.data_dir + os.sep + 'pickle_' + str(length) + os.sep + allele.replace('*','.').replace(':','_')+'.p', 'rb'))

            print('train on allele: ' + data_dict['allele'])
            if not length == data_dict['sequ_length']:
                print('length error')
                exit()

            encode_channel = data_dict['channel_encode']
            meas = data_dict['label']
            bind = []
            for i in meas:
                i = (-1) * math.log10(i);
                bind.append(i)
            sequ, label = encode_channel, bind

            if (len(sequ) > 0):
                sequ_ori, label_ori = sequ, label

                sequ_ori, test_sequ_ori, label_ori, test_label_ori = train_test_split(sequ_ori, label_ori, test_size=0.1, random_state=42, shuffle=True)


                output_list=[]
                label_list=[]

                fold_num = 0

                kf = KFold(n_splits=5, shuffle=True, random_state=42)

                for train_set, test_set in kf.split(sequ_ori, label_ori):

                    fold_num += 1

    #                alleleLoc = args.savedir + os.sep + allele + '.txt'

                    if os.path.isfile(alleleLoc):
                        log = open(alleleLoc, 'a')
                        log.write("\n")
                        log.write("%s\t\t\t%s\n" % ('Length: ', length))
                        log.write("%s\t\t\t\t%s\t\t\t%s\t\t\t%s\n" % ('Epoch', 'tr_loss', 'val_loss', 'val_Pearson'))
                        log.flush()
                    else:
                        log = open(alleleLoc, 'w')
                        log.write("%s\t\t\t%s\n" % ('Allele', allele))
                        log.write("\n")
                        log.write("%s\t\t\t%s\n" % ('Length: ', length))
                        log.write("%s\t\t\t\t%s\t\t\t%s\t\t\t%s\n" % ('Epoch', 'tr_loss', 'val_loss', 'val_Pearson'))
                        log.flush()


                    train_sequ, test_sequ, train_label, test_label = [sequ_ori[i] for i in train_set], [sequ_ori[i] for i in test_set],\
                                                                [label_ori[i] for i in train_set], [label_ori[i] for i in test_set]

                    train_sequ, val_sequ, train_label, val_label = train_test_split(train_sequ, train_label, test_size=0.1, random_state=42, shuffle=True)

                    train_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(train_sequ, train_label), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

                    val_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(val_sequ, val_label), batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)

                    test_data_load = torch.utils.data.DataLoader(myDataLoader.MyDataset(test_sequ, test_label),
                                                                 batch_size=args.batch_size, shuffle=True,
                                                                 num_workers=args.num_workers, pin_memory=True)

                    model = net.ResNetC1()

                    if args.onGPU == True:
                        #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
                        model = model.cuda()

#                    pretrain = torch.load('pretrain/best_model/' + allele + '/' + allele + '_' + str(length) + '.pth')
#                    model_dict = model.state_dict()
#                    pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
#                    model_dict.update(pretrained_dict)
#                    model.load_state_dict(model_dict)
                  
                    criteria = MSELoss()

                    if args.onGPU == True:
                        criteria = criteria.cuda()

                   # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

                    if args.onGPU == True:
                        cudnn.benchmark = True

                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.1)

                    start_epoch = 0

                    min_val_loss = 100
                    loss_not_decay = 0

                    for epoch in range(start_epoch, args.max_epochs):
                        tr_epoch_loss, tr_mean_squared_error = train(args, train_data_load, model, criteria, optimizer)
                        val_epoch_loss, val_mean_squared_error, val_output, val_label = val(args, val_data_load, model, criteria)

                        val_Pearson = pearsonr(val_output, val_label)

                        log = open(alleleLoc, 'a')
                        log.write("%s\t\t\t\t%.4f\t\t\t%.4f\t\t\t\t%.4f\n" % (epoch, tr_epoch_loss, val_epoch_loss, val_Pearson[0]))
            
                        if val_epoch_loss < min_val_loss:
                            if args.save_model == True:
                                model_file_name = model_dir + os.sep + allele + '_' + str(length) + '_' + str(fold_num) + '.pth'
                                print('==> Saving the best model')
                                torch.save(model.state_dict(), model_file_name)
                            min_val_loss = val_epoch_loss
                            loss_not_decay = 0
                        else:
                            loss_not_decay += 1

                        if loss_not_decay >= 40:
                            break

                        scheduler.step(epoch)

                    best_model_dict = torch.load(model_dir + os.sep + allele + '_' + str(length) + '_' + str(fold_num) + '.pth')
                    model.load_state_dict(best_model_dict)
                    _, _, output, label = val(args, test_data_load, model, criteria)

                    output_list.extend(output)
                    label_list.extend(label)

                pearson = pearsonr(output_list, label_list)
                r2 = r2_score(label_list, output_list)

                logger.write("%s\t%s\t\t\t\t%.4f\n" % (length, allele, pearson[0]))
                logger.flush()

    logger.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_path', default="iedb.csv")
    parser.add_argument('--data_dir', default="./data/")  # data directory
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--step_loss', type=int, default=5)
    parser.add_argument('--lr', type=float, default= 0.0005)
    parser.add_argument('--savedir', default='./results_vsepr_new')
    parser.add_argument('--crossValFile', default='crossValFile.txt')
    parser.add_argument('--visualizeNet', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--onGPU', default=True)

    #np.random.seed(42)
    
    cross_val(parser.parse_args())










