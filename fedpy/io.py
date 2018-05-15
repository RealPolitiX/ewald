# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd

# File I/O
def readcif(filename, **kwds):
    """
    :Parameters:
        filename : string
            filename address
    
    :Returns:
        atomLabels : string list
            atomic labels
        coords : ndarray
            atomic coordinates
        crystVec : list of numericals
            unit cell parameters in a cif file    
    """
    
    # Read the unit cell parameters
    a, b, c, alf, bet, gam = [[]]*6
    with open(filename, 'r') as f:
        
        for line in f:
            if "length_a" in line:
                a = numgrab(line)
            elif "length_b" in line:
                b = numgrab(line)
            elif "length_c" in line:
                c = numgrab(line)
            elif "angle_alpha" in line:
                alf = numgrab(line)
            elif "angle_beta" in line:
                bet = numgrab(line)
            elif "angle_gamma" in line:
                gam = numgrab(line)
    
    crystVec = a + b + c + alf + bet + gam
    
    # Read atomic coordinates
    cifdata = pd.read_csv(filename, delim_whitespace=True, header=None, **kwds)
    atomLabels = np.array(cifdata.values[:,0], dtype='str')
    coords = np.array(cifdata.values[:,1:4]).astype('float64')

    return atomLabels, coords, crystVec


def writexyz(atoms, coords, iteraxis, filename):
    """
    Write xyz file from coordinates
    
    :Parameters:
        atoms : list
            Atom list of strings
        coords : numpy array
            Atomic coordinates
        iteraxis : int
            axis to iterate
        filename : str
            filename
    """
    
    f = open(filename+'.xyz', 'w')
    
    nstruct = coords.shape[iteraxis]
    natom = len(atoms)
    coords = np.rollaxis(coords, iteraxis)
    
    for i in range(nstruct):
        f.write(str(natom)+'\n\n')
        for atom, coord in zip(atoms, coords[i,...]):
            f.write("{}   {} {} {}\n".format(atom, coord[0], coord[1], coord[2]))
    
    f.close()