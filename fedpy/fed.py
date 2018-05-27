# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd
import itertools as it
import .utils as u
pi = np.pi

# ==================================================
# Sections:
# 1.  Classes
# 2.  Conversions
# 3.  Simulation building blocks
# 4.  Simulation routines
# 5.  Basis decomposition
# ==================================================

# ============== Classes ============== #

class Detector(object):
    
    # Detector parameters
    def __init__(self, imgSizeX, imgSizeY, centreX, centreY, distance, pixelSize, magnification):
        
        self.imgSizeX = imgSizeX
        self.imgSizeY = imgSizeY
        self.centreX = centreX
        self.centreY = centreY
        self.dist = distance
        self.pixelSize = pixelSize
        self.magnification = magnification

        
class ElectronBeam(object):
    
    # Electron beam parameters
    def __init__(self, xyzFractionals, voltage, Crystal):
        
        self.dirFractionals = xyzFractionals
        dirCart = frac2cart(xyzFractionals,Crystal.axes)
        self.dirCart_norm = dirCart/np.linalg.norm(dirCart)
        self.dirReciprocal = cart2frac(self.dirCart_norm,Crystal.axesRecip)
        self.wavelength = voltage2wavelength(voltage)
        self.k0 = self.dirCart_norm/self.wavelength
        self.modK = 1.0/self.wavelength
        # Calculate orthonomal basis for e-beam: x || to crystal a-axis
        axes = np.zeros([3,3])
        axes[2] = self.dirCart_norm  # e-beam in z-direction
        vecOrthog_zx = np.cross(dirCart,Crystal.axes[0,:])  # direction orthogonal to e-beam and a-axis
        axes[1] = vecOrthog_zx/np.linalg.norm(vecOrthog_zx)
        axes[0] = np.cross(axes[1],axes[2])
        self.axes = axes


class TriclinicCrystal(object):
    
    def __init__(self, a, b, c, alpha, beta, gamma):
        
        # Basic geometric parameters
        deg2rad = np.pi/180.0
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        alpha = alpha*deg2rad
        beta = beta*deg2rad
        gamma = gamma*deg2rad
        V = a*b*c*((1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2 + \
                    2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))**0.5)
        
        # Transformation matrix and its inverse
        T = np.zeros([3,3])
        T[0,:] = [a, b*np.cos(gamma), c*np.cos(beta)]
        T[1,:] = [0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)]
        T[2,:] = [0, 0, V/(a*b*np.sin(gamma))]
        self.T_uvw2xyz = T
        self.axes = np.transpose(T)
        M = np.linalg.inv(T)
        self.T_xyz2uvw = M
        
        # Reciprocal space axis vectors
        A = T[:,0]
        B = T[:,1]
        C = T[:,2]
        CP = np.cross(B,C)   
        Astar = CP/np.dot(A,CP)
        CP = np.cross(C,A)
        Bstar = CP/np.dot(B,CP)
        CP = np.cross(A,B)
        Cstar= CP/np.dot(C,CP)
        self.axesRecip = np.array([Astar, Bstar, Cstar])
        
    def __repr__(self):
        
        return("Triclinic crystal\n\
                a = {},\n\
                b = {},\n\
                c = {},\n\
                alpha = {} deg,\n\
                beta = {} deg,\n\
                gamma = {} deg".format(self.a, self.b, self.c,\
                self.alpha, self.beta, self.gamma))


# ============== Conversions ============== #

def voltage2wavelength(voltage):
    """C onvert acceleration voltage of the electron
    to its de Broglie wavelength (in Angstrom)
    """
    
    h = 6.626069E-34
    e = 1.602176E-19
    me = 9.10938E-31
    c = 2.99792458E8
    term1 = h/np.sqrt(2*me*e*voltage)
    term2 = 1.0/np.sqrt(1 + (e*voltage/(2*me*c*c)))
    eWavelength = term1*term2*1e10 # unit in Angstrom
    
    return eWavelength

def frac2cart(xyzFrac,axes):
    
    xyzCart = np.dot(xyzFrac,axes)
    
    return xyzCart

def hkl2cart(hkl,axesRecip):
    
    xyzCart = np.dot(hkl,axesRecip)
    
    return xyzCart

def cart2frac(xyzCart,cellAxes):
    
    invAxes = np.linalg.inv(cellAxes)
    xyzFrac = np.array(np.dot(xyzCart,invAxes),'float64')
    
    return xyzFrac

def atomLabels2numbers(labels):
    
    nAt = labels.size
    atomicNumbers = np.zeros(nAt)
    for atm in range(nAt):
        if labels[atm] == 'H':
            atomicNumbers[atm] = 1
        elif labels[atm] =='C':
            atomicNumbers[atm] = 6
        elif labels[atm] == 'N':
            atomicNumbers[atm] = 7
        elif labels[atm] == 'I':
            atomicNumbers[atm] = 53
        else:
            print('atom {} not found'.format(atm))
    
    return atomicNumbers


# ============== Simulation building blocks ============== #

KirkTableMat = np.loadtxt(r'E:\Code-archiv\Python\PyNotebooks\TBAT\Simulation\KirklandTable.mat')
def calcAtomicF(Z,modQ):  # modQ = s/(2*pi)

    L0 = int(3*Z-3)
    A = np.array([KirkTableMat[L0,0],KirkTableMat[L0,2],KirkTableMat[L0+1,0]]).reshape(1,3)
    B = np.array([KirkTableMat[L0,1],KirkTableMat[L0,3],KirkTableMat[L0+1,1]]).reshape(3,1)
    C = np.array([KirkTableMat[L0+1,2],KirkTableMat[L0+2,0],KirkTableMat[L0+2,2]]).reshape(1,3)
    D = np.array([KirkTableMat[L0+1,3],KirkTableMat[L0+2,1],KirkTableMat[L0+2,3]]).reshape(3,1)
    q2 = np.tile(modQ**2,(3,1))   
    E = 1.0/(q2 + B)
    term1 = np.dot(A, E)
    term2 = np.dot(C, np.exp(-q2*D))
    atmF = term1 + term2
    
    return atmF

def calcStructureFactors(HKL, AtomicF, XYZ, form="complete"):
    """
    Calculate the complex-valued structure factor (SF)
    
    :Parameters:
        HKL : n x 3 array
            List of n (h,k,l) index triplets
        AtomicF : n x m array
            Atomic structure factor array (m atoms in a unit cell for the n index triplets)
        XYZ : m x 3 array
            Real-space fractional coordinates of all atoms (within unit cell)
        form : str
            Numerical form of the structure factor
    
    :Return:
        F : 1D array (n elements)
            Complex-valued structure factor for all index triplets
    """
    
    if form == "cos":
        sinf, cosf = 0, 1
    elif form == "sin":
        sinf, cosf = 1, 0
    elif form == "complete":
        sinf, cosf = 1, 1
    
    HKLdotXYZ = HKL.dot(XYZ.T)
    PhaseMatrix = np.cos(2*pi*HKLdotXYZ)*cosf + 1j*np.sin(2*pi*HKLdotXYZ)*sinf
    SF = np.einsum('ij->i', AtomicF*PhaseMatrix)
    
    return SF

def calcScatteringVectorsIMG(Detect,eBeam,rotation):
    
    nX = Detect.imgSizeX
    nY = Detect.imgSizeY
    ranX = np.arange(nX, dtype='float64')
    ranY = np.arange(nY, dtype='float64')
    qVectorIMG = np.zeros([nY,nX,3], dtype='float64')
    q = np.zeros([nY,nX,3], dtype='float64')
    modQ_IMG = np.zeros([nY,nX], dtype='float64')
    X, Y = np.meshgrid(ranX, ranY)
    dx = Detect.pixelSize*(X-Detect.centreX)/Detect.magnification
    dy = Detect.pixelSize*(Y-Detect.centreY)/Detect.magnification
    # calculate dR & theta matrix (element-wise)
    dR = np.sqrt(dx**2 + dy**2)
    theta = np.arcsin(dR/Detect.dist) # theta is a symmetric matrix
    # calculate phi matrix
    phi = np.where(dx==0, np.sign(dy)*np.pi/2, np.arctan(dy/dx)+np.pi*(1-np.sign(dx))/2) + rotation
    modK = 1.0/eBeam.wavelength
    modQ_IMG = 2*modK*np.sin(0.5*theta.T)
    qR = 2*modQ_IMG*np.cos(0.5*theta) # projection of q on detector
    q[:,:,0] = qR*np.cos(phi) # x-component of q
    q[:,:,1] = qR*np.sin(phi) # y-component of q
    q[:,:,2] = -2*modQ_IMG*np.sin(0.5*theta) # z-component of q (electron beam is -z direction)    
    #Transform to reference of crystal
    qVectorIMG = np.einsum('ijk,kl', q, eBeam.axes)
    #qVectorIMG = np.array([[np.dot(q[i,j,:],eBeam.axes) for i in ranY] for j in ranX])
    
    return qVectorIMG, modQ_IMG

def hkl2fAtomicAllAtoms(atomicNumbers, hklArray, axesReciprocal):
    """
    Calculate the atomic structure factor for all (h,k,l)
    
    :Parameters:
        atomicNumbers : 1D array
            Atomic numbers of the 
        hklArray : ndarray
            Array of (h,k,l) for the diffraction peaks
        axesReciprocal : 3x3 array
            Reciprocal space vectors of the unit cell
        
    :Return:
        fAtomic : ndarray
            Matrix of atomic structure factors
    """
    
    nAtoms = np.size(atomicNumbers)
    atSeq = np.arange(nAtoms)
    nHKL = np.size(hklArray, 0)

    # Initialize the atomic structure factor matrix
    fAtomic = np.zeros([nHKL, nAtoms])
    qVec = np.dot(hklArray, axesReciprocal) # nx3 2D array
    modQ = np.linalg.norm(qVec, axis=1) # 1D array
    
    for atm in atSeq:
        fAtomic[:, atm] = calcAtomicF(atomicNumbers[atm], modQ)
    
    # print('\nCalculated f_atomic for all ' + str(nAtoms) + ' atoms\n')
    return fAtomic

def calcHKLvaluesRequired(qVecIMG,axesReciprocal):
    
    hklExtremes = np.zeros([4,3])
    row = 0
    for X in [0,1]:
        for Y in [0,1]:
            A = [0,-1]            
            hklExtremes[row,:] = cart2frac(qVecIMG[A[Y],A[X],:],axesReciprocal)
            row = row+1
    print('h,k,l values at corners of detector:\n {}'.format(hklExtremes))
    HKLmaxVals = np.zeros(3)
    HKLminVals = np.zeros(3)
    HKLmaxVals = np.round(np.max(hklExtremes,0))+2
    HKLminVals = np.round(np.min(hklExtremes,0))-2
    print('maximum h,k,l values : {}'.format(HKLmaxVals))
    print('minimum h,k,l values : {}'.format(HKLminVals))
    Hrange = range(int(HKLminVals[0]),int(HKLmaxVals[0]))
    Krange = range(int(HKLminVals[1]),int(HKLmaxVals[1]))
    Lrange = range(int(HKLminVals[2]),int(HKLmaxVals[2]))
    HKL = np.zeros([np.size(Hrange)*np.size(Krange)*np.size(Lrange),3])
    row = 0
    for H in Hrange:
        for K in Krange:
            for L in Lrange:
                HKL[row] = [H,K,L]
                row = row+1
    return HKL

def removeUnnecessaryHKLs(HKLvals,eBeam,axesReciprocal,cutoff):
    # eBeam is an instance of the above ElectronBeam class
    # cutoff : thickness of reciprocal space to sample as ratio of |k0|
    
    k0norm = eBeam.dirCart_norm
    nHKL = np.size(HKLvals,0)
    qHKL = np.dot(HKLvals,axesReciprocal)
    HKLminimal = []
    for indxH in range(nHKL):
        distFromPlane = abs(np.dot(qHKL[indxH],k0norm))
        if distFromPlane < cutoff*eBeam.modK:
            HKLminimal.append(HKLvals[indxH])
    HKL2 = np.array(HKLminimal)
    
    return HKL2

def calcFhkl(hkl, atomCoords, atomicNos, axesReciprocal):
    
    # Calculate structure factors
    fAtomic_HKL = hkl2fAtomicAllAtoms(atomicNos, hkl, axesReciprocal)
    fractionals = cart2frac(atomCoords, axesReciprocal)
    Fhkl = calcStructureFactors(hkl, fAtomic_HKL, fractionals)
    
    return Fhkl


# ============== Simulation routines ============== #

# def sim_displaced():

#   pass


# ============== Basis decomposition ============== #

def lincompose(lincoeffs, components, mat_shape=None):
    """ Linear composition of bases
    """
    
    linmodel = np.dot(lincoeffs, components)
    
    try:
        matrecon = linmodel.reshape(mat_shape)
    except:
        matrecon = linmodel
    
    return matrecon
    
def polycompose(lincoeffs, components, axis=0, order=2, mat_shape=None):
    """ Polynomial composition of linear bases
    """
    
    nlincoeffs = kron2d(lincoeffs, axis=axis, order=order)
    nlincomps = kron2d(components, axis=axis, order=order)
    polymodel = np.dot(nlincoeffs, nlincomps)
    
    try:
        matrecon = polymodel.reshape(mat_shape)
    except:
        matrecon = polymodel
    
    return matrecon
    
def kron2d(mat, axis=0, order=2):
    """ Calcuate the multinomial feature vectors
    """
    
    if np.ndim(mat) == 1:
        mat = mat[:, np.newaxis]
    mat = np.rollaxis(mat, axis)
    nvec, nfeatlen = mat.shape
    
    # List the cross terms
    crosst = list(it.combinations(range(nvec), 2))
    # List the diagonal terms
    cardt = [(i, i) for i in range(nvec)]
    
    nterms = u.cnr(order+nvec-1, nvec-1)
    matkron = np.zeros((nterms, nfeatlen))
    
    # Calculate the diagonal terms
    matkron[:nvec, :] = mat**2
    # Calculate the cross terms
    for i, term in enumerate(crosst):
        matkron[nvec+i, :] = 2*mat[term[0], :]*mat[term[1], :]
    
    matkron = np.rollaxis(matkron, axis)
    
    return matkron.squeeze()