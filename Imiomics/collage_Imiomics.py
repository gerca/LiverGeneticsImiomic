  
import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import getopt
import pandas as pd
import sys
import glob
import helper_collage
from joblib import Parallel, delayed
import multiprocessing
import time


def main(argv):
    """
    Provide a collage for Imiomics maps
    Parameters
    ----------
    - i : input folder where to look for Imiomics maps and to later save the so-called collage
        Required structure: 
            Folder
               - male
                   Imiomics male maps (Jac,FF). Regression maps start with "beta", correlation maps start with "corr"
               - female
                   Imiomics female maps (Jac,FF). Regression maps start with "beta", correlation maps start with "corr"
               - place where maps will be stored
   
    - f : folder with MRI and mask data
         Required structure:
             Folder
                - small : Imiomics reference couple with small size
                    - female
                        - water image
                        - fat image
                        - bodymask
                    - male
                        - water image
                        - fat image
                        - bodymask
                - etc
                
    - s : size of Imiomics couple to be read
        ("small", "s" or "S" for small; "orig", "o", "original" for original; "l", "L" or "large" for large, 
        "T2D", "diab", or "diabetic" for T2D; "median_new" for the new median, "median_shape" for the shape median)
        Default: original
    - v : variable(s) name(s) for metadata used in Imiomics
         If they are several variables, please input them separated by commas
    - m : multiple comparison correction (MCC) method to be applied (optional). As statistical tests are computed
          voxelwise, there are multiple executions of statistical tests. Thus, there can be errors induced by the 
          multiple comparisons. There are different MCC methods that can be applied: "holm" for Holm s 
          multiple correction method, "hochberg" for Hochberg s method, and "rft" for Random Field Theory
    Many helper functions are used by this script, in "helper_collage.py"
    Returns
    -------
    Outputted Imiomics maps collages as "variable_collage.png", in the Imiomics folder (the one given as -i argument)
    """
    
    init = time.time()
    i = "" # Folder with Imiomics data
    f = "" # Folder with MRI and mask data
    v = "" # Variable names, spearated by commas (if more than one variable)
    s = "o" # Size of Imiomics couple (default: original couple)
    m = None # Multiple Comparison Correction method (default: None)

    try:
        opts, args = getopt.getopt(argv, "h:i:f:s:v:m:", ["imiomics=","folder=","size=","var=","mcc="])
    except getopt.GetoptError:
        print("collage_Imiomics.py -i <Imiomics folder> -f <folder> -s <size> -v <var> -m <mcc>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("collage_Imiomics.py -i <Imiomics folder> -f <data folder> -s <size> -v <var> -m <mcc>")
            sys.exit()
        elif opt in ("-i", "--imiomics"):
            i = arg
        elif opt in ("-f", "--folder"):
            f = arg        
        elif opt in ("-s", "--size"):
            s = arg
        elif opt in ("-v", "--var"):
            v = arg
        elif opt in ("-m", "--mcc"):
            m = arg

    
    if i == "" or f == "":
        print("collage_Imiomics.py -i <Imiomics folder> -f <folder> -s <size> -v <var> -m <mcc>")
        sys.exit()
    
    if not(os.path.exists(i)) or not(os.path.exists(f)) or os.path.isfile(i) or os.path.isfile(f):
        print("Imiomics folder {} or data folder {} do not exist or are not a directory".format(i,f))
        sys.exit()
        
    if v == "":
        print("No variable names provided, setting them by default from 'var1' to 'varN´ (N: #covariates)")
        
    if m is None:
        print("No Multiple Comparison Correction method defined. Skipping it...")
 
    
    # Get files of interest
    group_files,tag = helper_collage.getImiomicsPaths(i)
    mr_files,mask_files = helper_collage.getMRIMaskPaths(f,s)
    
    # load MRIs, and masks for each group
    mris,unmixed_mris = helper_collage.MRILoader(mr_files)
    masks = helper_collage.maskLoader(mask_files)
    
    # Get axial slice coordinates for collage
    axial_slices = helper_collage.axialSlices(mris[0],s)
    
    # Colormap information
    cmap = np.asarray([[128, 0, 0],  # Maroon
    
                               [255, 0, 0],  # Red
    
                               [255, 165, 0],  # Orange
    
                               [255, 255, 255],  # White
    
                               [0, 255, 255],  # Cyan
    
                               [0, 0, 255],  # Blue
    
                               [0, 0, 128]])  # Deep blue
    
    # Process input variable names
    if "," in v and v != "":
        v = v.split(",")    
    elif not("," in v) and v != "":
        v = [v]
    else:
        v = ["var{}".format(i) for i in range(1,len(group_files)+1)]
    
    # Parallel computation of different collages
    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(obtainCollage)(group,var,i,tag,mris,unmixed_mris,masks,axial_slices,cmap,m) for group,var in zip(group_files,v))
    
    # Remove outputted colorbar images during main collage computation
    helper_collage.removeColorbarImages(i)
    
    print("Analysis done! Time ellapsed: {}sec".format(round(time.time()-init,2)))


def obtainCollage(group,var,i,tag,mris,unmixed_mris,masks,axial_slices,cmap,m):
    """
    Function for parallellization of main collage computation
    Parameters
    ----------
    group : list of lists of str
        Set of Imiomics maps to be displayed
    var : list of str
        Set of variable names
    i : str
        Output folder where to leave maps
    tag : str
        Type of maps that are being analyzed ("beta" for regression, "corr" for correlation, "perc" for percentage increase)
    mris : list of np.ndarray
        Loaded MR images for background of main collage (mixed Water and Fat signals)
    unmixed_mris : list of np.ndarray
        Loaded MR images for background of main collage (mixed Water and Fat signals)
    masks : list pf np.ndarray
        Loaded mask images for background of main collage
    axial_slices : list of ints
        Axial coordinates where axial plots are being plotted in the collage
    cmap : np.ndarray
        Colormap applied in collage
    m : str or None
        Method for Multiple Comparison Correction
    Returns
    -------
    None.
    """
    
    try:
        # load maps for each group  
        maps, extrema, n = helper_collage.mapLoader(group,masks,m)
        
        # Get axial, coronal and sagittal slices
        axial_maps,axial_mri,axial_masks,coronal_maps,coronal_mri,coronal_masks,sagittal_mri,sagittal_masks = helper_collage.getSlices(maps,mris,masks,axial_slices)
        
        # Get curves for right-hand side plots
        curves,curve_extrema = helper_collage.curveLines(maps,masks,unmixed_mris)
        
        axial_components = [axial_maps,axial_mri,axial_masks]
        coronal_components = [coronal_maps,coronal_mri,coronal_masks]
        sagittal_components = [sagittal_mri,sagittal_masks]
        
        # Get Colorbar plots
        colorbar_images =  helper_collage.colorbarPlot(cmap,extrema,var,i,tag)
        
        # Mix maps with masks and MRI
        mixed_axial,mixed_coronal,mixed_sagittal = helper_collage.paintSlices(axial_components,
                                                                              coronal_components,sagittal_components,
                                                                              cmap,extrema)
    
        # Inputs for figure Layout: axial_maps,axial_mri,axial_masks,coronal_maps,coronal_mri,coronal_masks,sagittal_mri,sagittal_masks
        helper_collage.figureLayout(mixed_axial,mixed_coronal,mixed_sagittal,colorbar_images,curves,curve_extrema,axial_slices,var,n,i)

    except: # Avoid faulty logistic regression maps
        print("No significant voxels found for '{}'. Skipping...".format(var))

        
if __name__ == '__main__':
    main(sys.argv[1:])
