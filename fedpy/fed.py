# -*- coding: utf-8 -*-

from __future__ import division
from . import utils as u
import numpy as np
import pandas as pd
import numba as nb
import itertools as it
import scipy.ndimage.filters as spfilters
pi = np.pi
fwhm2sigma = 2*np.sqrt(2*np.log(2))

# ==================================================
# Sections:
# 1.  Classes
# 2.  Conversions
# 3.  Simulation building blocks
# 4.  Simulation routines
# 5.  Basis decomposition
# 6.  Loss functions
# ==================================================

# ============== Classes ============== #

class Detector(object):
    """ Detector parameter container class
    """

    def __init__(self, imgSizeX, imgSizeY, centreX, centreY, distance, pixelSize, magnification):

        self.imgSizeX = imgSizeX
        self.imgSizeY = imgSizeY
        self.centreX = centreX
        self.centreY = centreY
        self.dist = distance
        self.pixelSize = pixelSize
        self.magnification = magnification

    def __repr__(self):

        return("Detector parameters:\n\
                X, Y = {}, {}\n\
                Xcenter, Ycenter = {}, {}\n\
                Detector distance = {},\n\
                Pixel size = {},\n\
                Magnification = {}".format(self.imgSizeX, self.imgSizeY, \
                self.centreX, self.centreY, self.dist, self.pixelSize, \
                self.magnification))


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

    def xyz2uvw(self, xyz):

        xyzT = np.transpose(xyz)
        uvwT = np.dot(self.T_xyz2uvw,xyzT)
        uvw = np.transpose(uvwT)

        return uvw

    def uvw2xyz(self, uvw):

        uvwT = np.transpose(uvw)
        xyzT = np.dot(self.T_uvw2xyz,uvwT)
        xyz = np.transpose(xyzT)

        return xyz


# ============== Conversions ============== #

#@nb.njit("float64(float64)")
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

def frac2cart(xyzFrac, axes):
    """
    Conversion from fractional to Cartesian coordinates
    """

    xyzCart = np.dot(xyzFrac,axes)

    return xyzCart

def hkl2cart(hkl, axesRecip):

    xyzCart = np.dot(hkl,axesRecip)

    return xyzCart

def cart2frac(xyzCart, cellAxes):
    """
    Conversion from Cartesian to fractional coordinates
    """

    invAxes = np.linalg.inv(cellAxes)
    xyzFrac = np.array(np.dot(xyzCart,invAxes),'float64')

    return xyzFrac

def atomLabels2numbers(labels):
    """
    Convert atomic symbols to atomic number
    """

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

KirkTableMat = np.loadtxt(r'.\KirklandTable.mat')
def calcAtomicF(Z, modQ):  # modQ = s/(2*pi)

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

def npcalcStructureFactors(HKL, AtomicF, XYZ, form="complete"):
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

    nHKL, _ = HKL.shape
    HKLdotXYZ = np.dot(HKL, XYZ.T) # n x m array
    PhaseMatrix = np.cos(2*pi*HKLdotXYZ)*cosf + 1j*np.sin(2*pi*HKLdotXYZ)*sinf # n x m array
    #SF = np.einsum('ij->i', AtomicF*PhaseMatrix)
    SF = np.zeros((nHKL,))
    for ind in nb.prange(nHKL):
        SF[ind] = np.sum(AtomicF[ind,:]*PhaseMatrix[ind,:])

    return SF

@nb.njit("complex128[:](float64[:,:], float64[:,:], float64[:,:], int64)")
def nbcalcStructureFactors(HKL, AtomicF, XYZ, form=2):
    """
    Calculate the complex-valued structure factor (SF)

    :Parameters:
        HKL : n x 3 array
            List of n (h,k,l) index triplets
        AtomicF : n x m array
            Atomic structure factor array (m atoms in a unit cell for the n Miller-indexed triplets)
        XYZ : m x 3 array
            Real-space fractional coordinates of all atoms (within unit cell)
        form : str
            Numerical form of the structure factor

    :Return:
        F : 1D array (n elements)
            Complex-valued structure factor for all index triplets
    """

    nHKL, _ = HKL.shape
    HKLdotXYZ = np.dot(HKL, XYZ.T) # n x m array

    if form == 0: # Cosine
        PhaseMatrix = np.cos(2*pi*HKLdotXYZ) + 0j
    elif form == 1: # Sine
        PhaseMatrix = 1j*np.sin(2*pi*HKLdotXYZ)
    elif form == 2: # Complete
        PhaseMatrix = np.cos(2*pi*HKLdotXYZ) + 1j*np.sin(2*pi*HKLdotXYZ)

    # Calculate the structure factor (SF)
    SF = np.zeros((nHKL,), dtype=np.complex128)
    for ind in nb.prange(nHKL):
        SF[ind] = np.sum(AtomicF[ind,:]*PhaseMatrix[ind,:])

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

@nb.njit
def calcPhi(dx, dy):
    """
    Calculate the arctangent of the complex number in the range [-pi/2, pi/2]
    """

    if dx == 0:
        PHI = np.pi*0.5
        if dy < 0:
            PHI = -PHI
    else:
        PHI = np.arctan(dy/dx)
        if dx < 0:
            PHI = PHI+np.pi

    return PHI

@nb.njit
def calcDeltaK(dx, dy, rotation, wavelen, axes, detdist):
    """
    Calculate the theta angle for a given K vector
    """

    q = np.zeros(3)
    dR = np.sqrt(dx**2 + dy**2)
    theta = np.arcsin(dR/detdist)
    phi = calcPhi(dx, dy) + rotation
    modK = 1.0/wavelen
    hsinth = np.sin(0.5*theta)
    modQ = 2*modK*hsinth
    q[2] = 2*modQ*hsinth # z-component of q (electron beam is -z direction)
    qR = 2*modQ*np.cos(0.5*theta) # projection of q on detector
    q[0] = qR*np.cos(phi) # x-component of q
    q[1] = qR*np.sin(phi) # y-component of q
    #Transform to reference of crystal
    Q = np.dot(q, axes)

    return Q, modQ

def nbcalcScatteringVectorsIMG(Detect,eBeam,rotation):
    """ Calculate the scattering vector (jit-version)
    """

    nX, nY = Detect.imgSizeX, Detect.imgSizeY
    qVectorIMG = np.zeros([nY, nX, 3])
    modQ_IMG = np.zeros([nY, nX])

    for X in nb.prange(nX):
        for Y in nb.prange(nY):
            dx = Detect.pixelSize*(X-Detect.centreX)/Detect.magnification
            dy = Detect.pixelSize*(Y-Detect.centreY)/Detect.magnification
            qVectorIMG[Y,X,:], modQ_IMG[Y,X] = calcDeltaK(dx, dy, rotation, eBeam.wavelength, eBeam.axes, Detect.dist)

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

    HKLmaxVals = np.zeros(3)
    HKLminVals = np.zeros(3)
    HKLmaxVals = np.round(np.max(hklExtremes,0))+2
    HKLminVals = np.round(np.min(hklExtremes,0))-2

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

    return np.array(HKLminimal)

def calcFhkl(hkl, atomCoords, atomicNos, axesReciprocal):

    # Calculate structure factors
    fAtomic_HKL = hkl2fAtomicAllAtoms(atomicNos, hkl, axesReciprocal)
    fractionals = cart2frac(atomCoords, axesReciprocal)
    Fhkl = calcStructureFactors(hkl, fAtomic_HKL, fractionals)

    return Fhkl

def convoluteIMGandBeam(IMG, beamSizeFWHM, detectionProperties):
    sigma = beamSizeFWHM / (detectionProperties.pixelSize*fwhm2sigma)
    imgconv = spfilters.gaussian_filter(IMG, sigma)
    return imgconv

@nb.njit("float64[:](float64[:], float64[:])", parallel=True)
def nbcross(u, v):
    # Cross product between two 1D vectors (w = u x v)

    w = np.zeros_like(u)
    w[0] = u[1]*v[2] - u[2]*v[1]
    w[1] = u[2]*v[0] - u[0]*v[2]
    w[2] = u[0]*v[1] - u[1]*v[0]

    return w

@nb.njit("float64[:,:](float64[:,:,:], float64[:,:], complex128[:], float64[:], float64, float64, float64)", parallel=True)
def calcDiffractionPatternwithMosaicity(qIMG, qHKL, Fhkl, k0, thetaMosaicParallel, thetaMosaicRotational, excitationError):
    """ Faster version of calcDiffractionPattern_ParallelRotationalMosaic()
    """

    Ihkl = (Fhkl*np.conj(Fhkl)).real
    modK = np.linalg.norm(k0)
    k0norm = k0/modK

    nY, nX, _ = qIMG.shape
    IMG = np.zeros((nY, nX))

    excite2rd, thetaMosaicRot2rd, thetaMosaicPar2rd = np.array([excitationError, thetaMosaicRotational, thetaMosaicParallel])**2
    normFactorRadial = 1/excitationError

    for Y in nb.prange(nY):

        for X in nb.prange(nX):

            # Calculate intensities from all hkls at each pixel
            # Calculate the q vector at every detector pixel position
            qPixel = qIMG[Y,X,:]
            modQpix = np.sqrt(qPixel[0]**2 + qPixel[1]**2 + qPixel[2]**2)
            qPixelNorm = qPixel / modQpix
            qOrthNorm = nbcross(k0norm, qPixelNorm)

            dQ = qPixel - qHKL # nHKLx3 matrix, dQ vector for each hkl trio
            mod_dQ = np.sqrt(dQ[:,0]**2 + dQ[:,1]**2 + dQ[:,2]**2) # nHKL array

            # Calculate exponential coefficients for the mosaicity model
            dQradial2rd = np.abs(np.dot(dQ, qPixelNorm))**2 # nx1 array
            factorRadial = 0.5*dQradial2rd / excite2rd

            dQrot = np.dot(dQ, qOrthNorm) # nHKLx1 array
            sigmaRotate2rd = modQpix*modQpix*thetaMosaicRot2rd + excite2rd
            factorRotate = 0.5*dQrot*dQrot / sigmaRotate2rd

            dQlong2rd = np.dot(dQ, k0norm)**2 # nHKLx1 array
            sigmaLong2rd = modQpix*modQpix*thetaMosaicPar2rd + excite2rd
            factorLong = 0.5*dQlong2rd / sigmaLong2rd

            expFactor = np.exp(-factorRadial-factorRotate-factorLong) * Ihkl / np.sqrt(sigmaRotate2rd*sigmaLong2rd)
            Ixy = (mod_dQ < 0.02*modK)*expFactor # nHKLx1 array
            IMG[Y, X] = np.sum(Ixy) * normFactorRadial

    return IMG


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
    """ Calculate the multinomial feature vectors
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

# ============== Loss functions ============== #

def linear_unmix_loss(coeffs, Sexp, Scomps, A):
    """
    Loss function for linear unmixing
    """

    # coeffs -- coefficients
    # Sexp -- vectorized experimental difference map
    # Scomps -- composite matrix made of unmixing bases

    Smodel = np.dot(A, Scomps)
    loss = np.sum(Sexp.ravel() - Smodel)**2

    return loss

def nonlinear_unmix_loss(coeffs, Sexp, Scomps, A, order=2):
    """
    Loss function for nonlinear unmixing
    """

    linloss = linear_unmix_loss(coeffs, Sexp, Scomps, A)
    polydiff = (Sexp.ravel())**order - polycompose(coeffs, Scomps)
    polyloss = np.sum(polydiff**2)

    loss = linloss + polyloss

    return loss

def simloss(a, diffdata, dispmat, xyzgs, imgoff, mask, atoms, bd):
    """
    Loss function for simulated-based optimization
    """

    totaldisp = np.dot(dispmat, a).reshape(xyzgs.shape)
    on_structure = xyzgs + totaldisp

    imgon = ps.simulate2(atoms, xyzgs, on_structure, mag=1.11, rot=137, beamsize=300e-6, exrat=0.2, \
                         mosL=2.0, mosR=1.0, beam_direction=-bd, method='rotmosaic', rebin=(2,2))
    diffsim = 7e-6*(imgon - imgoff)*mask

    loss = np.linalg.norm(diffdata - diffsim)**2

    return loss
