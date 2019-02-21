# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:45:18 2018

@author: jtfl2
"""

import generateAminoAcids as gAA
import numpy as np
import sys
import os
import time
import easygui as eg
#from flask import Flask
aa = 'A-ALA,ASN,ASP,CYS,GLN,GLU,GLY,HID,HIE,HIP,HIS,ILE,LEU,LYS,MET,PHE,PRO,R-ARG,SER,THR,TRP,TYR,VAL'.split(',')



path2 = os.getcwd()
path1 = ''
for i in path2:
    if i == '\\':
        path1 = path1+'/'
    else:
        path1 = path1+i

os.chdir(path1 + '/PDBFILES')
def getAVG():
    for i in aa:
        filename = i + '.pdb'
        gAA.generateAmino(filename, getAVG = True)
os.chdir(path1)

def normalizeData(peptide):
    if os.path.isfile('allAtoms.txt') and os.path.isfile('allAngles.txt') and os.path.isfile('allBondLength.txt') and os.path.isfile('allBondAmounts.txt') and os.path.isfile('allLonePairs.txt'):
        allAtoms = np.loadtxt('allAtoms.txt', dtype = 'float').tolist()
        allAngles = np.loadtxt('allAngles.txt',  dtype = 'float').tolist()
        allBondLength = np.loadtxt('allBondLength.txt', dtype = 'float').tolist()
        allBondAmounts = np.loadtxt('allBondAmounts.txt', dtype = 'float').tolist()
        allLonePairs = np.loadtxt('allLonePairs.txt', dtype = 'float').tolist()
        for i in range(int(peptide.shape[0]/5)):
            j = i*5
            peptide[j,0] = (peptide[j,0]- np.mean(allAtoms))/np.std(allAtoms)
            peptide[j,1] = (peptide[j,1]- np.mean(allLonePairs))/np.std(allLonePairs)
            for k in range(peptide.shape[1]):
                if peptide[j+1,k] != 0:
                    peptide[j+1,k] = (peptide[j+1,k]- np.mean(allBondAmounts))/np.std(allBondAmounts)
                if peptide[j+2,k] != 0:
                    peptide[j+2,k] = (peptide[j+2,k]- np.mean(allAtoms))/np.std(allAtoms)
                if peptide[j+3,k] != 0:
                    peptide[j+3,k] = (peptide[j+3,k]- np.mean(allBondLength))/np.std(allBondLength)
                if peptide[j+4,k] != 0:
                    peptide[j+4,k] = (peptide[j+4,k]- np.mean(allAngles))/np.std(allAngles)
        return peptide
                
    else:
        getAVG()
        normalizeData(peptide)

def pepEncoder(peptides = [], cT = 1, saved = '', ftype = '', aminopath = path1 + '/PDBFILES' , useNormal = False, csv = True, picture = True):
    asked  = False
    max_length = 0
    max_width = 0   
    j = 0
    
    
    if len(peptides) == 0:
        ftype = eg.buttonbox('Type in a single sequence, or read from csv file?', choices = ('Sequence','Csv file'))
        if ftype.lower() == 'sequence':
            ftype = 's'
            peptides = [list(eg.enterbox('Sequence: ').upper())]
        elif ftype.lower() == 'csv file':
            ftype = 'c'
            filename = ''
            filenamelist = list(eg.fileopenbox(msg = 'Csv file location:'))
            for i in filenamelist:
                if i != '\\':
                   filename = filename + i
                else:
                   filename = filename +'/'
            peptides = np.loadtxt(filename, dtype = 'str')
        else:
            con = eg.ynbox('Please give a valid response, \n Continue? ')
            if con:
                pepEncoder()
            else:
                sys.exit('exit')
        savedlist = list(eg.diropenbox(msg = 'Where did you want this to be saved?'))
        cT = int(eg.buttonbox('Is this with 1 character amino acids, or 3?', choices = ('1','3')))
        for i in savedlist:
                if i != '\\':
                   saved = saved + i
                else:
                   saved = saved +'/'
        useNormal = eg.ynbox('Did you want to nomralize the ecoding?')
        savetype = eg.buttonbox('Save it as a picture, csv, or both?', choices = ('Picture', 'Csv file', 'Both'))
        if savetype == 'Picture':
            csv = False
        if savetype == 'Csv file':
            picture = False
    if saved == '':
        saved = path1 + '/peptides'
    if not os.path.exists(saved):
        os.makedirs(saved)
    peplen = len(peptides) 
    times = [] 
    counting = False                           
    for i in peptides:
        j = j + 1 
        if saved != '':
            os.chdir(saved)
        if os.path.isfile('peptide'+str(j) +'.csv') and not asked:
            rewrite = eg.ynbox(msg = "Rewrite current peptides?")
            asked = True
        if os.path.isfile('peptide'+str(j) +'.csv') and not rewrite:
            matrix = np.genfromtxt('peptide'+str(j) +'.csv', delimiter=',')
            max_length = max(max_length, matrix.shape[0])
            max_width = max(max_width, matrix.shape[1])
        else:
            begin = time.time()
            if ftype == 'c':
                sequence = i.upper().split(',')
            else:
                sequence = i
            if aminopath != '':
                os.chdir(aminopath)
            matrix = gAA.generatePep(sequence, charType= cT)
            max_length = max(max_length, matrix.shape[0])
            max_width = max(max_width, matrix.shape[1])
            if saved != '':
                os.chdir(saved)
            gAA.printSequence(matrix, 'peptide'+str(j), makecsv = csv, makepicture = picture)    
            end = time.time()
            times.append((end-begin)/3600)
        if peplen > 50:
            if counting or len(times) != 0:
                counting = True
                str_format = "{0:." + str(5) + "f}"
                print ("Encoding: ","ETA: ", str(round((sum(times)/len(times)*(peplen-j)), 4)), "(hours)", str_format.format(100 * (j / float(peplen))),"percent complete         ", end = '\r')
            else:
                str_format = "{0:." + str(5) + "f}"
                print ("Encoding: ","ETA: "," ??? " , "(hours)", str_format.format(100 * (j / float(peplen))),"percent complete         ", end = '\r')
                
    print ('')
    
    counting = False
    j = 0
    times = []
    end = 0
    for filename in os.listdir(saved):
        if filename.endswith(".csv"):
            begin = time.clock()
            j = j + 1
            matrix = np.genfromtxt(filename, delimiter=',')
            if useNormal:
                if aminopath != '':
                    os.chdir(aminopath)
                matrix = normalizeData(matrix)
                if saved != '':
                    os.chdir(saved)
            filename = filename.replace(".csv", "")
            m = np.zeros((max_length, max_width))
            m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix
            gAA.printSequence(m, filename, makecsv = csv, makepicture = picture)
            end = time.clock()
            times.append((end-begin)/3600)
        if peplen > 50:
            if counting or len(times) != 0:
                counting = True
                str_format = "{0:." + str(5) + "f}"
                print ("Reformating: ","ETA: ", str(round((sum(times)/len(times)*(peplen-j)), 4)), "(hours)", str_format.format(100 * (j / float(peplen))),"percent complete         ", end = '\r')  
            else:
                str_format = "{0:." + str(5) + "f}"
                print ("Reformating: ","ETA: "," ??? " , "(hours)", str_format.format(100 * (j / float(peplen))),"percent complete         ", end = '\r')

    print ('Sucessfully genorated ', str(peplen), ' peptides.')
    for i in reversed(range(10)):
        print ('Closing in: ', str(i+1), end = '\r')
        time.sleep(1)
    return

#app = Flask(__name__)
#@app.route('/')
##def home():
##    return render_template('calc.html')
#
#def index():
#    print(pepEncoder())
#    return
#
#if __name__ == "__main__":
#    app.run()