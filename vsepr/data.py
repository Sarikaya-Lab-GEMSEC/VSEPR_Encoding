# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:34:42 2018

@author: lux32, jtfl2
"""
import numpy as np
import pandas as pd
import os
import math
aa = list('AILMVFWYNCQSTDERHKGP')

def make_matrix (sequence):
    matrix = np.zeros((20, 13))
    sequence = list(sequence)
    print(sequence)
    for i in range(len(sequence)):
        for j in range(len(aa)):
            if sequence[i] == aa[j]:
                matrix[j, i] = 1
    return np.reshape(matrix, (20, 1, 13))


def load_data(file_path):
    csv_path = os.path.join(file_path, "iedb.csv")
    data = pd.read_csv(csv_path)
    data = data.loc[data['species'] == 'human']
    data = data.loc[(data['peptide_length'] == 9) | (data['peptide_length'] == 10)]
    #data = data.loc[data['mhc'] =='HLA-A*01:01']
    #print(data.head())
    sequ = data['sequence'].values.tolist()
    matrix_list = [make_matrix(x) for x in sequ]
    meas = data['meas'].values.tolist()
    bind = []
    for i in meas:
        i = (-1) * math.log10(i);
        bind.append(i)
    return matrix_list, bind


