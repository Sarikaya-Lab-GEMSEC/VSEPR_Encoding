# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:31:56 2018

@author: jtfl2
"""

import pandas as pd
import generatePep as gP
import os
from time import time
mhc = []
path1 = os.getcwd()
df = pd.read_table('bdata.20130222.mhci.public.1.txt')
allel = df.loc[df['species'] == 'human']['mhc']
for i in allel:
    if i not in mhc:
        mhc.append(i)

maxL = 0
maxW = 0
print("Making all of the peptides")
sequences = df.loc[df['species'] == 'human']
for j in mhc:
    now = time();
    currentSeq = sequences.loc[sequences['mhc'] == j]['sequence']
    seq = []
    for i in currentSeq:
        if list(i) not in seq:
            seq.append(list(i))
            print(''.join(list(i)), end = '\r')
#            gP.make_pdb(''.join(list(i)), path1 + '/bdata.human.withnames.PDBdividedByMHC/' + j.replace('*','.').replace(':','_'))
    [L, W] = gP.pepEncoder(peptides = seq, saved = path1 + '/benchmark/' + j.replace('*','.').replace(':','_'), useNormal=True, picture = False, getmaxsize = True) 
    maxL = max(maxL,L)
    maxW = max(maxW, W)
    print("\n Current overall Max Size: (" +str(maxL) + ',' + str(maxW) + ')')
    print(j,(time()-now())/60)
print("Making all of them the same size")          
for j in mhc:
    now = time();
    currentSeq = sequences.loc[sequences['mhc'] == j]['sequence']
    seq = []
    for i in currentSeq:
        if list(i) not in seq:
            seq.append(list(i))
            print(''.join(list(i)), end = '\r')
#            gP.make_pdb(''.join(list(i)), path1 + '/bdata.human.withnames.PDBdividedByMHC/' + j.replace('*','.').replace(':','_'))
    [maxL, maxW] = gP.pepEncoder(peptides = seq, saved = path1 + '/benchmark/' + j.replace('*','.').replace(':','_'), useNormal=True, picture = False, max_length=maxL, max_width=maxW)
    print(j,(time()-now())/60)
