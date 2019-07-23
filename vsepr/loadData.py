import numpy as np
import os.path
import pickle
import pandas as pd

class loadData:
    def __init__(self, data_dir, csv_path, sequ_length, allele): # sequ_length 9 or 10
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.sequ_length = sequ_length
        self.allele = allele
        self.encode = list()
        self.channel_encode = list()
        self.label = list()


    def readFlile(self):
        label_file = pd.read_csv(self.csv_path)
        for folder in os.listdir(self.data_dir):
            if folder == self.allele:
                for file in os.listdir(self.data_dir + os.sep + folder):
                    if file.endswith('.csv') and len(file.split('.')[0]) == self.sequ_length:
                        print(self.allele + ': ' + file.split('.')[0])
                        encode_matrix = list()
                        channel_matrix = [[], [], [], [], []]
                        encode_file = pd.read_table(self.data_dir + os.sep + folder + os.sep + file, header=None).as_matrix()
                        for (i, line) in enumerate(encode_file):
                            encode_list = line[0].split(',')
                            encode_list = [float(x) for x in encode_list]
                            encode_matrix.append(encode_list)
                            channel_matrix[i%5].append(encode_list)
                        self.encode.append(np.asanyarray(encode_matrix))
                        self.channel_encode.append(np.asanyarray(channel_matrix))

                        csv_allele = allele.split('.')[0] + '*' + allele.split('.')[1].split('_')[0] + ':' + allele.split('.')[1].split('_')[1]
                        label = label_file.loc[(label_file['sequence'] == file.split('.')[0]) & (label_file['mhc'] == csv_allele)]['meas'].values.item()
                        self.label.append(label)
        return 0

    def processData(self):
        if not os.path.exists('data' + os.sep + 'pickle_9'):
            os.mkdir('data' + os.sep + 'pickle_9')
        if not os.path.exists('data' + os.sep + 'pickle_10'):
            os.mkdir('data' + os.sep + 'pickle_10')
        return_val = self.readFlile()

        if return_val == 0:
            print('Pickling data')

            data_dict = dict()
            data_dict['allele'] = self.allele
            data_dict['sequ_length'] = self.sequ_length
            data_dict['encode'] = self.encode
            data_dict['channel_encode'] = self.channel_encode
            data_dict['label'] = self.label

            if self.sequ_length == 9:
                pickle.dump(data_dict, open('data' + os.sep + 'pickle_9' + os.sep + self.allele + '.p', "wb"))
            elif self.sequ_length == 10:
                pickle.dump(data_dict, open('data' + os.sep + 'pickle_10' + os.sep + self.allele + '.p', "wb"))
            else:
                print('Invalid sequence length!!')
            return data_dict
        return None



interest_allele = ['HLA-A.02_01','HLA-A.03_01']
#interest_allele = ['HLA-A.02_01','HLA-A.03_01','HLA-A.11_01','HLA-A.11_01','HLA-A.02_03','HLA-B.15_01','HLA-A.31_01','HLA-A.01_01','HLA-B.07_02','HLA-A.26_01',
                   #'HLA-A.02_06','HLA-A.68_02','HLA-B.08_01','HLA-B.58_01','HLA-B.40_01','HLA-B.27_05','HLA-A.30_01','HLA-A.69_01','HLA-B.57_01','HLA-B.35_01',
                   #'HLA-A.02_02','HLA-A.24_02','HLA-B.18_01','HLA-B.51_01','HLA-A.29_02','HLA-A.68_01','HLA-A.33_01','HLA-A.23_01']
for allele in interest_allele:
    loadData("./data/encoding", "./data/iedb.csv", 9, allele).processData()
    loadData("./data/encoding", "./data/iedb.csv", 10, allele).processData()

test = pickle.load(open('data/pickle_9/HLA-A.03_01.p', "rb"))
print(test['label'])




