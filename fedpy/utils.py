# -*- coding: utf-8 -*-

import re
import numpy as np
from math import factorial as fac

def numgrab(string, delim=' '):
    """Find the numerical values in a string
    """
    
    strspl = re.split(delim, string)
    numlist = []
    for s in strspl:
        try:
            numlist.append(float(s))
        except:
            pass
    
    return numlist
    
def arraybin2(arr, rowbin=1, colbin=1):
    """
    Binning a 2D array
    
    :Parameters:
        arr : numpy array
            array to bin
        rowbin : int | 1
            row binning
        colbin : int | 1
            column binning
        
    :Return:
        arrbinned : numpy array
            binned array
    """
    
    oldr, oldc = arr.shape
    newr, newc = oldr//rowbin, oldc//colbin
    shape = (newr, rowbin, newc, colbin)
    arrbinned = arr.reshape(shape).mean(-1).mean(1)
    
    return arrbinned
    
def cnr(n, r):
    """ Calculate the combinatorial coefficient
    """
    
    coef = fac(n) / (fac(r) * fac(n-r))
    
    return int(coef)