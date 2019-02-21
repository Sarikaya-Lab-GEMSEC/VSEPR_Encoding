# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:41:03 2018

@author: jtfl2
"""

import prody as pd
import itertools as it
import makematrix as mm
import numpy as np
from PIL import Image
import os
from operator import itemgetter
#import sys


char3 = 'A-ALA,ASN,ASP,CYS,GLN,GLU,GLY,HID,HIE,HIP,HIS,ILE,LEU,LYS,MET,PHE,PRO,R-ARG,SER,THR,TRP,TYR,VAL'.split(',')
char1 = 'A,N,D,C,Q,E,G,Hid,Hie,Hip,H,I,L,K,M,F,P,R,S,T,W,Y,V'.split(',')

# Establishes all of the peptides at the begining rather than doing them when needed.
# This in the long run becomes more effiecent
path2 = os.getcwd()
path1 = ''
for i in path2:
    if i == '\\':
        path1 = path1+'/'
    else:
        path1 = path1+i
        
parsedPDB = []
for i in char3:
    parsedPDB.append(pd.parsePDB(path1 + '/PDBFILES/' + i + '.pdb'))

# Encodes a single animo Acid given the pdb file for the AA. 
# can be iterated through with generatePep inorder to make full string peptides with interlocking atoms
def generateAmino(filename, order = 1, c_term = False, getAVG = False):
    if getAVG:
        avgpath = path1 + '/avgInfo/'
        if os.path.isfile(avgpath + 'allAtoms.txt') and os.path.isfile(avgpath + 'allAngles.txt') and os.path.isfile(avgpath + 'allBondLength.txt') and os.path.isfile(avgpath + 'allBondAmounts.txt') and os.path.isfile(avgpath + 'allLonePairs.txt'):
            allAtoms = np.loadtxt(avgpath + 'allAtoms.txt', dtype = 'float').tolist()
            allAngles = np.loadtxt(avgpath + 'allAngles.txt',  dtype = 'float').tolist()
            allBondLength = np.loadtxt(avgpath + 'allBondLength.txt', dtype = 'float').tolist()
            allBondAmounts = np.loadtxt(avgpath + 'allBondAmounts.txt', dtype = 'float').tolist()
            allLonePairs = np.loadtxt(avgpath + 'allLonePairs.txt', dtype = 'float').tolist()
        else:
            allAtoms = []
            allAngles = []
            allBondLength = []
            allBondAmounts = []
            allLonePairs = []
    bondinfopath = path1 + '/BondInfo/'
    if os.path.isfile(bondinfopath + filename + 'found_bonds.txt') and os.path.isfile(bondinfopath + filename + 'bond_index.txt'):
        found_bonds = np.loadtxt(bondinfopath + filename + 'found_bonds.txt', dtype = 'str').tolist()
        bond_index = np.loadtxt(bondinfopath + filename + 'bond_index.txt', dtype = 'int').tolist()
    else:
        found_bonds = []
        bond_index = []
    i = parsedPDB[char3.index(filename.replace('.pdb',''))]
    neighbors = pd.measure.findNeighbors(i,1.9)
    hasAngle = []
    found = []
    for j in neighbors:
        pairs = [j[0], j[1]]
        for atom in pairs:
            if atom not in found and list(str(atom))[5] != 'H':
                current = []
                current.append(atom)
                for k in neighbors:
                    if atom == k[0]:
                        current.append(k[1])
                    elif atom == k[1]:
                        current.append(k[0])
                if len(current) >= 3:
                    hasAngle.append(current)
                    found.append(atom)
    core_atoms = []                
    for k in range(len(found)):
        core_atoms.append((k, found[k].getName()))
    core_atoms = sorted(core_atoms, key=itemgetter(1))
    oldHasAngle = hasAngle
    hasAngle = []
    for k in core_atoms:
       hasAngle.append(oldHasAngle[k[0]]) 
    
    [row,col,base] = mm.makemajormatrix(np.zeros((0,0)), 100)
    for p in range(len(found)):
        l = hasAngle[p]
        angles = []
        index = "".join(str(x) for x in range(1,len(l)))
        o = it.product(index, '0', index)
        usedCombos = []
        for combo in o:
            combo = [int(x) for x in combo]
            if sorted(combo, key=int)  not in usedCombos and combo[0] != combo[2]: 
                a = pd.measure.calcAngle(l[combo[0]], l[combo[1]], l[combo[2]], radian = False)
                angles.append([a, l[combo[0]], l[combo[2]]])   
                usedCombos.append(sorted(combo, key=int))
        bond_type = []
        bond_lengths = []
        bond_angles = []
        usedCombos = []
        for x in angles:
            name1 = x[1].getName()
            name2 = x[2].getName()
            bond_type.append(name1)
            bond_type.append(0)
            bond_type.append(name2)
            bond_lengths.append(pd.measure.calcDistance(l[0], x[1]))
            bond_lengths.append(0)
            bond_lengths.append(pd.measure.calcDistance(l[0], x[2]))
            bond_angles.append(0)
            bond_angles.append(x[0])
            bond_angles.append(0)
            
        bond_amounts = []
            
        center = l[0].getName()
        if center[0] == 'C':
            e_needed = 4                        
        if len(center) > 1 and center[0] == 'H' and center[1] == 'X' and center[2] == 'T':
            e_needed = 6
        if center[0] == 'O':
            e_needed = 6
        if center[0] == 'N':
            e_needed = 5
        if center[0] == 'S':
            e_needed = 6
            
        
        
        for m in range(1,len(l)):
            if l[m].getName()[0] == 'H':
                bond_amounts.append(1)
            elif center == 'CA':
                bond_amounts.append(1)
            elif center == 'C' and l[m].getName() == 'O':
                bond_amounts.append(2)
            elif l[m].getName() == 'C' and center == 'O':
                bond_amounts.append(2)    
            elif sorted([center, l[m].getName()]) not in found_bonds:
                print(filename)
                print(center + ' bond with '  + l[m].getName())
                n = input()
                bond_index.append(n)
                bond_amounts.append(n)
                found_bonds.append(sorted([center, l[m].getName()]))
            else:
                bond_amounts.append(bond_index[found_bonds.index(sorted([center, l[m].getName()]))])
        
        remaining_e = (e_needed - sum(bond_amounts))
        lp = 0
        charge = 0
        while remaining_e > 0:
            if remaining_e == 1:
                charge = -1
            if remaining_e == -1:
                charge = 1
            if remaining_e >= 2:
                lp = lp + 1
            remaining_e = remaining_e - 2
            
            
        matrix = mm.makeatom(l[0].getName(),lp, charge, bond_amounts, bond_type,  bond_lengths, bond_angles, order, c_term)
        [row, col, base, n] = mm.addmatrix(base, row, col, matrix, 'down')
        if getAVG:
            allLonePairs.append(lp)
            for k in bond_amounts:
                if k != 0:
                    allBondAmounts.append(k)
            for k in bond_lengths:
                if k != 0:
                    allBondLength.append(k)
            for k in bond_angles:
                if k != 0:
                    allAngles.append(k)
            allAtoms.append(matrix[0,0])
            for k in matrix[2,:]:
                if k != 0:
                    allAtoms.append(k)
            
        
    if c_term:
        [row,col, base, n] = mm.addmatrix(base, row, col, mm.makeatom('HXT', 2, 0, [1,1],['H',0,'C'],[.957,0,1.364],[0,106.1,0], order, True), 'down')
    np.savetxt(filename.replace('.pbd','') + 'found_bonds.txt', found_bonds,fmt='%s')
    np.savetxt(filename.replace('.pbd','') + 'bond_index.txt', bond_index, fmt = '%s')
    if getAVG:
         np.savetxt('allAtoms.txt', allAtoms,fmt='%s')
         np.savetxt('allAngles.txt', allAngles,fmt='%s')
         np.savetxt('allBondLength.txt', allBondLength,fmt='%s')
         np.savetxt('allBondAmounts.txt', allBondAmounts,fmt='%s')
         np.savetxt('allLonePairs.txt', allLonePairs,fmt='%s')

    return mm.cleanmatrix(base)
        
def generatePep(sequencelist, charType = 1):
    size = 100
    seql = []
    if charType == 1:
        first = True
        for i in sequencelist:
            if first:
                seql.append(char3[char1.index(i[-1])])
                first = False
            else:
                seql.append(char3[char1.index(i[0])])
    elif charType == 3:
        for i in sequencelist:
            seql.append(i)
    [row, col, current] = mm.makemajormatrix(np.zeros((0,0)), 100)
    for j in range(len(seql)):
        if j == len(seql) - 1:
            c_term = True
        else:
            c_term = False
        filename = seql[j] + '.pdb'
        add = generateAmino(filename, j, c_term)
        [row, col, current, size] = mm.addmatrix(current, row, col, add, 'down', n = size)
    return mm.cleanmatrix(current)

def printSequence(matrix, filename, makecsv = True, makepicture = True):
    if makecsv:
        np.savetxt(filename + '.csv', matrix, delimiter=",")
    if makepicture:
        im = Image.fromarray(matrix, 'RGB')
        im.save(filename + '.png')
        
    
    
    
    
    
    
    
    
    
    
    
    