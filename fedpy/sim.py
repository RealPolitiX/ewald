# -*- coding: utf-8 -*-

import numpy as np
from . import utils as fu, fed as fed

binning = 4
imageSizeX = int(320/binning)
imageSizeY = int(320/binning)
cenX = imageSizeX*0.5-0.51
cenY = imageSizeY*0.5-0.51
detDist = 0.5
pixelSize = binning*24e-6

voltage = 1.1e5
wavelength = fed.voltage2wavelength(voltage)
modk = 1/wavelength

def simulate(atomList, XYZs, **kwargs):
    """
    Simulation of a single-component diffraction pattern
    """

    magnification = kwargs.pop('mag', 0.7)
    beamSize = kwargs.pop('beamsize', 250e-6)
    DetectionParameters = fed.Detector(imageSizeX, imageSizeY, cenX, cenY, detDist, pixelSize, magnification)
    rotation = kwargs.pop('rot', 90)
    rotation = rotation*np.pi/180.0
    B = kwargs.pop('B', 0.04)
    DW = kwargs.pop('DW', False)
    mosL = kwargs.pop('mosL', 1.0)
    mosR = kwargs.pop('mosR', 3.0)
    mosaicityLong = mosL*np.pi/180.0
    mosaicityRot = mosR*np.pi/180.0
    excitedRatio = kwargs.pop('exrat', 0.2)
    rebin = kwargs.pop('rebin', (1, 1)) # Rebinning factor in postprocessing

    # Beam direction in fractional coords
    beamDirection_Brotherton = np.array([-6.9942, 1.261112729, -4.308685046]) # Default beam direction
    beamdir = kwargs.pop('beam_direction', beamDirection_Brotherton)

    # Unit cell geom from Carole
    unitCell_opt = fed.TriclinicCrystal(9.217051, 15.571887, 15.940388, 83.9111, 73.34017, 78.294699)
    # Unit cell from Brotherton et al. 2012
    unitCell_Brotherton = fed.TriclinicCrystal(9.5161, 15.6549, 15.8860, 83.4610, 74.2290, 78.5220)
    eBeam_Brotherton = fed.ElectronBeam(beamdir, voltage, unitCell_Brotherton)

    # Calculate the scattering q vectors on the detector screen
    qVectors, modQ = fed.nbcalcScatteringVectorsIMG(DetectionParameters, eBeam_Brotherton, rotation)

    # Calculate the relevant (hkl) indices
    HKLall = fed.calcHKLvaluesRequired(qVectors, unitCell_Brotherton.axesRecip)
    HKL = fed.removeUnnecessaryHKLs(HKLall, eBeam_Brotherton, unitCell_Brotherton.axesRecip, 0.006)
    UVW = unitCell_opt.xyz2uvw(XYZs)

    # Retrieve the atomic number from the list
    atomicNumbers = fed.atomLabels2numbers(atomList)

    # Calculate substituent atomic contributions to all (hkl)
    fAtomic = fed.hkl2fAtomicAllAtoms(atomicNumbers, HKL, unitCell_Brotherton.axesRecip)

    # Calculate the structure factors for all (hkl)
    Fhkls = fed.nbcalcStructureFactors(HKL, fAtomic, UVW, form=0)

    # Calculate the q vectors for all (hkl)
    qHKL = fed.frac2cart(HKL, unitCell_Brotherton.axesRecip)

    if DW:
        # Calculate Debye-Waller factor
        Thhkl = fed.nbcalcHKLTheta(qHKL, eBeam_Brotherton.k0, magnification)
        DWhkl = fed.necalcDWFactor(Thhkl, wavelength, B)
        Fhkls *= DWhkl

    # Calculate diffraction pattern taken crystalline mosaicity into consideration
    img = fed.calcDiffractionPatternwithMosaicity(qVectors, qHKL, Fhkls, eBeam_Brotherton.k0, \
    mosaicityLong, mosaicityRot, modk*pixelSize/detDist)

    # Convolve the diffraction pattern with the electron beam shape
    img_conv = fed.convoluteIMGandBeam(img, beamSize, DetectionParameters)

    # Apply binning to the calculate diffraction pattern
    imgbin = fu.arraybin2(img_conv, rowbin=rebin[0], colbin=rebin[1])

    return imgbin

def simulate2(atomList, *XYZs, **kwargs):
    """
    Simulation of a two-component diffraction pattern
    """

    magnification = kwargs.pop('mag', 0.7)
    beamSize = kwargs.pop('beamsize', 250e-6)
    DetectionParameters = fed.Detector(imageSizeX, imageSizeY, cenX, cenY, detDist, pixelSize, magnification)
    rotation = kwargs.pop('rot', 90)
    rotation = rotation*np.pi/180.0
    Bg = kwargs.pop('Bg', 0.04)
    Be = kwargs.pop('Be', 0.04)
    DW = kwargs.pop('DW', False)
    mosL = kwargs.pop('mosL', 1.0)
    mosR = kwargs.pop('mosR', 3.0)
    mosaicityLong = mosL*np.pi/180.0
    mosaicityRot = mosR*np.pi/180.0
    excitedRatio = kwargs.pop('exrat', 0.2)
    rebin = kwargs.pop('rebin', (1, 1)) # Rebinning factor in postprocessing

    # Beam direction in fractional coords
    beamDirection_Brotherton = np.array([-6.9942, 1.261112729, -4.308685046])
    beamdir = kwargs.pop('beam_direction', beamDirection_Brotherton)

    # Unit cell geom from Carole
    unitCell_opt = fed.TriclinicCrystal(9.217051, 15.571887, 15.940388, 83.9111, 73.34017, 78.294699)
    # Unit cell from Brotherton et al. 2012
    unitCell_Brotherton = fed.TriclinicCrystal(9.5161, 15.6549, 15.8860, 83.4610, 74.2290, 78.5220)

    eBeam_Brotherton = fed.ElectronBeam(beamdir, voltage, unitCell_Brotherton)
    qVectors, modQ = fed.nbcalcScatteringVectorsIMG(DetectionParameters, eBeam_Brotherton, rotation)

    # Calculate the relevant (hkl) indices
    HKLall = fed.calcHKLvaluesRequired(qVectors, unitCell_Brotherton.axesRecip)
    HKL = fed.removeUnnecessaryHKLs(HKLall, eBeam_Brotherton, unitCell_Brotherton.axesRecip, 0.006)
    UVW_GS = unitCell_opt.xyz2uvw(XYZs[0])
    UVW_ES = unitCell_opt.xyz2uvw(XYZs[1])

    # Retrieve the atomic number from the list
    atomicNumbers = fed.atomLabels2numbers(atomList)

    # Calculate substituent atomic contributions to all (hkl)
    fAtomic = fed.hkl2fAtomicAllAtoms(atomicNumbers, HKL, unitCell_Brotherton.axesRecip)

    # Calculate the q vectors for all (hkl)
    qHKL = fed.frac2cart(HKL, unitCell_Brotherton.axesRecip)

    Fhkls_GS = fed.nbcalcStructureFactors(HKL, fAtomic, UVW_GS, form=0)
    Fhkls_ES = fed.nbcalcStructureFactors(HKL, fAtomic, UVW_ES, form=0)

    if DW:
        # Calculate Debye-Waller factor
        Thhkl = fed.nbcalcHKLTheta(qHKL, eBeam_Brotherton.k0, magnification, wavelength)
        DWhkl_GS = fed.necalcDWFactor(Thhkl, wavelength, Bg)
        DWhkl_ES = fed.necalcDWFactor(Thhkl, wavelength, Be)
        Fhkls_GS *= DWhkl_GS
        Fhkls_ES *= DWhkl_ES

    # Calculate the mixed structure factors for all (hkl)
    Fhkls = (1-excitedRatio)*Fhkls_GS + excitedRatio*Fhkls_ES

    # Calculate diffraction pattern taken crystalline mosaicity into consideration
    img = fed.calcDiffractionPatternwithMosaicity(qVectors, qHKL, Fhkls, eBeam_Brotherton.k0, \
    mosaicityLong, mosaicityRot, modk*pixelSize/detDist)

    # Convolve the diffraction pattern with the electron beam shape
    img_conv = fed.convoluteIMGandBeam(img, beamSize, DetectionParameters)

    # Apply binning to the calculate diffraction pattern
    imgbin = fu.arraybin2(img_conv, rowbin=rebin[0], colbin=rebin[1])

    return imgbin
