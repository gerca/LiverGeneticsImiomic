import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import sys
import itertools
import glob
from scipy.ndimage import center_of_mass,gaussian_filter,median_filter
from scipy.ndimage.morphology import binary_erosion
from skimage.measure import label
from scipy.stats import zscore
from matplotlib import cm
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap
import PIL.Image
import matplotlib as mpl
import math
import time


# Helper functions for the main Imiomics collage code

def getImiomicsPaths(folder):
    """
    Get main paths for Imiomics maps in given folder, in the order:
    male jacobian, female jacobian, male FF, female FF
    
    Params:
        - folder : str
            Folder where to look for Imiomics maps
        
    Outputs:
        - found_files : list of lists of str
            Found files of interest to plot collages: [[male_jac1,female_jac1,male_ff1,female_ff1],
                                                       [male_jac2,female_jac2,male_ff2,female_ff2],...]
        - t : str
            Type of map read, for later legend formatting
    
    """
    
    files = np.array(sorted(glob.glob(os.path.join(folder,"**","*.nrrd"),recursive=True))) # Look for all NRRD files
        
    if len(files) == 0:
        print("No NRRD files found in {}".format(folder))
        sys.exit()
    
    # Filter for files with "/beta", "/perc" or "/corr" tags
    tags = ["/beta","/corr","/perc"]
    indexes = [np.flatnonzero(np.core.defchararray.find(files,tag)!=-1) for tag in tags]
    indexes = list(itertools.chain.from_iterable(indexes))    
    imiomics_maps = files[indexes]
    
    # Get groups of Imiomics files
    found_files = []
    cont = 1
    while True:
        tags = ["/beta{}".format(cont), "/corr{}".format(cont),"/perc{}".format(cont)]
        found = 0
        for enum_tag,tag in enumerate(tags):
            tag_index = np.flatnonzero(np.core.defchararray.find(imiomics_maps,tag)!=-1)
            if len(tag_index) > 0: # Found Imiomics files, now sort them properly
                group_files = imiomics_maps[tag_index]
                group_files_lower = np.array([str(group_file).lower() for group_file in group_files])
                found += len(group_files)
                group_tags = [["jac","_m_"],["jac","_f_"],["fat","_m_"],["fat","_f_"]] # Order of files found
                group = []                
                for group_tag in group_tags:
                    index_tags = []
                    for image_tag in group_tag:
                        index_tag = np.flatnonzero(np.core.defchararray.find(group_files_lower,image_tag)!=-1)
                        index_tags.append(index_tag)
                    index_file = np.intersect1d(index_tags[0],index_tags[1])
                    group.append(str(group_files[index_file][0]))
                found_files.append(group)
                tag_found = enum_tag

        if found == 0:
            if tag_found == 0:
                t = "beta"
            elif tag_found == 1:
                t = "corr"
            elif tag_found == 2:
                t = "perc"

            return found_files,t
        else:
            cont += 1
                    

def getMRIMaskPaths(folder,size="o"):
    """
    Get paths of MRI data and masks for the collage

    Parameters
    ----------
    folder : str
        Folder where to look for MRI and mask path data
    size : str
        Size of Imiomics couple we are interested in (default: "o")

    Returns
    -------
    mri_paths : list of list of str
        Paths for MR images ([[male_mri_water,male_mri_fat],[female_mri_water,female_mri_fat]])
    mask_paths : list of str
        Paths for mask images ([male_mask_path, female_mask_path])

    """
    # Get subfolder where to look for
    if size.lower() == "o" or "orig" in size.lower():
        subfolder = "original" # Original Imiomics couple
    elif size.lower() == "s" or size.lower() == "small":
        subfolder = "small"
    elif size.lower() == "l" or size.lower() == "large":
        subfolder = "large"
    elif size.lower() == "t2d" or "diab" in size.lower():
        subfolder = "T2D"
    else:
        subfolder = size
        
    genders = ["male","female"]
    mask_paths = []
    mri_paths = []
    
    for gender in genders:
        
        full_folder = os.path.join(folder,subfolder,gender)
        if not os.path.exists(full_folder):
            print("Data folder {} does not exist".format(full_folder))
            sys.exit()
        files = np.array(sorted(os.listdir(full_folder)))
        files_lower = np.array([str(file).lower() for file in files])
        tags = ["_w.","_f.","mask"]
        inds = []
        for tag in tags:
            ind = np.flatnonzero(np.core.defchararray.find(files_lower,tag)!=-1)
            inds.append(ind[0])

        files = files[inds]
        mri_paths.append(files[:(-1)].tolist())
        mri_paths[-1] = [os.path.join(full_folder,mri_path) for mri_path in mri_paths[-1]]
        mask_paths.append(os.path.join(full_folder,str(files[-1])))
        
    return mri_paths,mask_paths


def getDataSize(files):
    """
    Get data size for what is being analyzed

    Parameters
    ----------
    files : list of str
        Imiomics files to be loaded

    Returns
    -------
    N : list of int
        Number of male and female subjects analyzed ([N male, N female])
    N_diab : list of int
        Number of male and female diabetic subjects analyzed ([N_diab male, N_diab female])

    """
    files = files[:2]
    N = []
    N_diab = []
    
    for enum_file,file in enumerate(files):
        split_point = file.split(".")[0]
        N.append(int(split_point.split("_")[-1]))
        if "diab" in file.lower() or "t2d" in file.lower():
            if enum_file == 0:
                N_diab.append(1083)
            elif enum_file == 1:
                N_diab.append(499)
                
    return N, N_diab


def euler_formula(z,dof):
    """
    Get formula for computing Euler characteristic in 3D, for Multiple 
    Comparison Correction with Random Field Theory

    Parameters
    ----------
    z : float
        Z statistic for comparison
    dof : int
        Number of degrees of freedom

    Returns
    -------
    euler_char : float
        Euler characteristic in 3D for the given Z level

    """
    
    euler_char = (((4*math.log(2))**(2/3))/((2*math.pi)**2))*(((dof-1)*(z**2))/(dof)-1)*(1+(z**2)/(dof))**(-0.5*(dof-1))
    
    return euler_char



def multipleComparison_correction(p_map,mask,method=None,alpha=0.05):
    """
    Apply Holm's method for Multiple Comparison Correction

    Parameters
    ----------
    p_map : np.ndarray
        P-value map
    mask : np.ndarray
        Corresponding bodymask
    method : str
        Multiple Comparison Correction method (default: "Holm", can be also "Hochberg", or "rft")
    alpha : float
        Significance level to be applied (default: 0.05)
 
    Returns
    -------
    corrected_mask : np.ndarray
        Significance mask obtained with Holm's method

    """
    mask_aux = np.copy(mask)
    mask_aux[mask > 0] = 1
    m = np.sum(mask_aux.flatten())
    
    # Filter arrays only on the body
    p_filtered = p_map[mask > 0]
    coord_mask = np.array(np.where(mask > 0))
    
    sort = np.argsort(p_filtered)
    p_filtered = p_filtered[sort]
    coord_mask = coord_mask[:,sort]
    
    if method is None:
         corr = alpha
    else:
        if method.lower() == "holm": # Correction criterion with Holm's method
            corr = alpha/(m+1-np.arange(len(p_filtered)))
        elif method.lower() == "hochberg": # Correction criterion with Hochberg's method
            corr = np.arange(len(p_filtered))*alpha/m
        elif method.lower() == "rft": # Correction criterion with Random Field Theory
            corr = euler_formula(1.64,3)


    ind = np.where(p_filtered < corr)[0]

    corrected_mask = np.zeros(p_map.shape)
    
    if len(ind) > 0: # Significant areas found
        significant_coord = coord_mask[:,ind]
        corrected_mask[significant_coord[0],significant_coord[1],significant_coord[2]] = 1
        
    return corrected_mask


def filter_pvalue(file,img,mask,method=None,level=0.05):
    """
    Try to filter Imiomics maps for p-value significance

    Parameters
    ----------
    file : str
        Imiomics map file that is being analyzed
    img : np.ndarray
        Array for Imiomics map
    mask : np.ndarray
        Array for bodymask
    method : str
        Multiple Comparison Correction (default: None, can be "holm","hochberg", or "rft")
    level : float
        Significance level used in p-value analysis

    Returns
    -------
    filtered_img : np.ndarray
        Filtered image according to p-value significance
        If no p-value map is found, a copy from the original image is given

    """
    
    # find p-value file
    child_file = os.path.basename(file)
    parent_folder = os.path.dirname(file)
    p_value_file = os.path.join(parent_folder,child_file.replace("beta","p")) # Regression
    
    if not(os.path.exists(p_value_file)) or (p_value_file == file):
        p_value_file = os.path.join(parent_folder,child_file.replace("corr","p")) # Correlation
        if not(os.path.exists(p_value_file)) or (p_value_file == file):
            p_value_file = os.path.join(parent_folder,child_file.replace("perc","p")) # Percentage increase           
            if not(os.path.exists(p_value_file)) or (p_value_file == file):
                return img
    
    
    # Some p-value file has been found
    p_img = sitk.GetArrayFromImage(sitk.ReadImage(p_value_file))
    p_img = np.flip(p_img,1)
    
    # Deal with NaN values of p-value images. Set them to 1
    p_img = np.nan_to_num(p_img,nan=1.0)
    
    p_img_mask = multipleComparison_correction(p_img,mask,method,level)
    
    # Remove noise from colormaps with median filter processing in longitudinal maps
    if "longitudinal" in p_value_file.lower():
        p_img_mask_filter = median_filter(p_img_mask,size=3)
        p_img_mask_filter *= p_img_mask

    return img*p_img_mask



def mapClipping(img):
    """
    Clip map to remove outliers

    Parameters
    ----------
    img : np.ndarray
        Imiomics map array to be clipped

    Returns
    -------
    clip_img : np.ndarray
        Imiomics map array that has been clipped
    clip : list of float/int
        Set of maxima and minima (or percentiles) used to clip

    """
    
    # Clip mainly with 1st and 99th percentiles
    clip = [np.percentile(img[img != 0].flatten(),1), np.percentile(img[img != 0].flatten(),99)]
    #clip = [np.amin(img), np.amax(img)]
    
    
    if clip[0]*clip[1] > 0 and clip[0] > 0:
        clip[0] = np.amin(img[img != 0])
    elif clip[0]*clip[1] > 0 and clip[1] < 0:
        clip[1] = np.amax(img[img != 0])
    
    # In cases with extrema close to zero    
    # if (np.amax(img) > 0) and (clip[1] < 0):
    #     clip[1] = np.amax(img)
    
    # if (np.amin(img) < 0) and (clip[0] > 0):
    #     clip[0] = np.amin(img)
        
    
    return np.clip(img,clip[0],clip[1]), clip
    

def mapLoader(files,masks,method=None):
    """
    Load Imiomics maps information

    Parameters
    ----------
    files : list of str
        Imiomics files to be loaded
    masks : list of np.ndarray
        Set of bodymasks
    mathod : None or str
        Multiple Comparison Correction method (default: None, can be "holm","hocheberg",or "rft")

    Returns
    -------
    maps : list of np.ndarray
        Loaded maps
    extrema : list of float/int
        Extreme values for Jac and FF ([[minimum_jac,maximum_jac],[minimum_ff,maximum_jac]])
    num : list of int
        Set of male and female datasizes used in the Imiomics map computation

    """
    maps = []
    extrema = []
    num = []
    
    for enum_file,file in enumerate(files):
        
        if enum_file%2 == 0: # define lists for saving the extrema values read
            maxima = []
            minima = []
            
        if enum_file < len(files)//2:
            file_split = file.split(".")[0]
            n = int(file_split.split("_")[-1])
            num.append(n)
        
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(file))
            img = np.flip(img,1)
        except:
            #pass
            print("Image file '{}' could not be opened".format(file))
            sys.exit()
        
        # Filter Imiomics map for significant values    
        img = filter_pvalue(file,img,masks[enum_file],method,0.05)
        
        # Clip Imiomics map between 1st and 99th percentiles
        img, clip = mapClipping(img)
        maxima.append(clip[1])
        minima.append(clip[0])
        
        if enum_file%2 == 1: # Get maxima and minima for each type of map
            infimum = min(minima)
            supremum = max(maxima)
            extrema.append([infimum,supremum])
        
        maps.append(img)
    
    return maps, extrema, num
        
        

def MRIMixer(imgs):
    """
    Get an average MRI image from the water and fat versions

    Parameters
    ----------
    imgs : list of np.ndarray
        Arrays to be mixed

    Returns
    -------
    mix : np.ndarray
        Mixed MR image

    """  
      
    return 0.5*(imgs[0] + imgs[1])


        
def MRILoader(files):
    """
    Load MRI files for collage

    Parameters
    ----------
    files : list of lists of str
        MRI files to be loaded: ([[male_mri_water,male_mri_fat],[female_mri_water,female_mri_fat]])

    Returns
    -------
    mri_img : list of np.ndarray
        Loaded MRI data and mixed: [mixed_male_MRI, mixed_female_MRI]*2
    unmixed_mri : list of lists of np.ndarray
        Loaded MRI data and unmixed: [unmixed_male_MRI, unmixed_female_MRI]*2

    """
    
    mri_img = []
    unmixed_mri = []
    
    for file in files:
        mris = []
        for typ in file:
            try:
                img = sitk.GetArrayFromImage(sitk.ReadImage(typ))
                img,_ = mapClipping(np.flip(img,1))
            except:
                print("MRI file {} could not be accessed".format(typ))
            mris.append(img)
        mri_img.append(MRIMixer(mris))
        unmixed_mri.append(mris)
    
    return mri_img*2,unmixed_mri*2   
        


def maskLoader(files):
    """
    Load masks

    Parameters
    ----------
    files : list of str
        Mask files to be loaded

    Returns
    -------
    masks : list of np.ndarray
        Loaded mask files as arrays

    """
    
    masks = []
    for file in files:
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(file))
            #img,_ = mapClipping(np.flip(img,1))
            img = np.flip(img,1)
        except:
            print("Mask file {} could not be accessed".format(file))
            
        masks.append(img)
    
    return masks*2


def axialSlices(img,size):
    """
    Provide slices where to apply the axial plots

    Parameters
    ----------
    img : np.ndarray
        Sample map where to get the size of the image
    size : str
        Size of Imiomics couple to be analyzed

    Returns
    -------
    axial_slices : list of np.ndarray
        List of axial slices to be analyzed

    """
    
    if size.lower() == "o" or "orig" in size.lower():
        male_axial_slices = np.array([285, 269, 237, 195, 142, 100]) # Original subject
        female_axial_slices = np.array([292, 269, 251, 228, 173, 129]) # Original subject
    elif size.lower() == "s" or size.lower() == "small":
        male_axial_slices = np.array([298, 275, 248, 198, 165, 85]) # Small subject
        female_axial_slices = np.array([308,285,265,225,188,125]) # Small subject
    elif size.lower() == "l" or size.lower() == "large":
        pass
    elif size.lower() == "t2d" or "diab" in size.lower():
        pass
    elif size.lower() == "median_shape":
        male_axial_slices = np.array([298, 275, 245, 192, 155, 72]) # Median-shape subject
        female_axial_slices = np.array([305,278,262,198,168,85]) # Median-shape subject
    elif size.lower() == "median_new":
        male_axial_slices = np.array([305,270,220,195,160,100]) # Median-new subject
        female_axial_slices = np.array([310,275,240,220,173,120]) # Median-new subject
        
    return [male_axial_slices,female_axial_slices]


def getCenterofMass(mask):
    """
    Get the coronal and sagittal centers of mass for the computation
    of mid-sagittal and mid-coronal projections for the collage

    Parameters
    ----------
    mask : np.ndarray
        Bodymask used in the processing

    Returns
    -------
    coronal_com : np.ndarray of float
        Coronal set of centers of mass
    sagittal_com : np.ndarray of float
        Sagittal set of centers of mass

    """  
    
    mask_aux = np.copy(mask)
    mask_aux[mask > 0] = 1 # Mask binarization
    
    coms = [] # Centers of mass
    
    for i in range(mask_aux.shape[0]):
        pos = center_of_mass(mask_aux[i])
        coms.append(list(pos))
    
    coms = np.array(coms).astype(float)
    
    return coms[:,0], coms[:,1]


def interpolate(img,coord,coord_coronal,axial_slice,mode="coronal",typ="map"):
    """
    Get interpolated coronal or sagittal view for the collage
    
    Params:
        - img : np.ndarray
            Image to be interpolated
        - coord : np.ndarray
            Set of centers of mass to be used during the interpolation
        - coord_coronal : np.ndarray
            Set of coronal centers of mass to be used during interpolation
        - axial_slice : np.ndarray
            Set of axial slices
        - mode : str
            View to be provided (default: "coronal")
        - typ : str
            Type of image analyzed ("map","MRI","mask")
    
    Returns:
        - slice2d : np.ndarray
            Interpolated image
    
    """
    
    if mode == "coronal":
        slice2d = np.empty((img.shape[0],img.shape[-1]),"f")
        for enum_c,c in enumerate(coord):
            int_coord = int(round(coord[enum_c]))
            #if (enum_c == 0) or (enum_c == len(coord) - 1): 
            slice2d[enum_c] = img[enum_c,int_coord,:]
            #else:
                # Interpolation with surrounding coronal slices
                #slice2d[enum_c] = img[enum_c,int_coord + 1,:]*(coord[enum_c] - math.floor(coord[enum_c])) + img[enum_c,int_coord - 1,:]*(math.ceil(coord[enum_c]) - coord[enum_c])
    
        if typ == "MRI":
            slice2d = gaussian_filter(slice2d,0.3)
    
    elif mode == "sagittal":
        slice2d = np.empty((img.shape[0],img.shape[1]),"f")
        axial_slice = axial_slice.tolist()
        for enum_c,c in enumerate(coord):
            int_coord = int(round(coord[enum_c]))
            #if (enum_c == 0) or (enum_c == len(coord) - 1): 
            slice2d[enum_c] = img[enum_c,:,int_coord]
            #else:
                # Interpolation with surrounding coronal slices
                #slice2d[enum_c] = img[enum_c,:,int_coord + 1]*(coord[enum_c] - math.floor(coord[enum_c])) + img[enum_c,:,int_coord - 1]*(math.ceil(coord[enum_c]) - coord[enum_c])
            
            slice2d[enum_c,int(round(coord_coronal[enum_c]))] = 0
            
            if enum_c in axial_slice:
                slice2d[enum_c] = np.zeros(slice2d.shape[-1])
                
        if typ == "MRI":
            slice2d = gaussian_filter(slice2d,0.3)
                
        slice2d = np.fliplr(slice2d)

    slice2d = np.flipud(slice2d)

    return slice2d


def getAxialSlice(img,slice_coord):
    """
    Extract axial slices for Imiomics collage

    Parameters
    ----------
    imgs : np.ndarray
        Map from where to extract the axial slices
    slice_coord : np.ndarray
        Axial slice coordinates to be extracted

    Returns
    -------
    slices : list of np.ndarray
        Set of axial slices to be used for further processing

    """
    
    slices = [] # List of lists with extracted axial slices
    img_aux = np.flip(img,1)
    for coord in slice_coord:
        slices.append(img_aux[coord])
        
    return slices




def getCoronalSlice(img,coronal_com,typ="map"):
    """
    Obtain coronal slice for collage
    
    Params:
        img : np.ndarray
            Image to be analyzed
        coronal_com : np.ndarray
            Set of coronal centers of mass
        typ : str
            Type of image to be analyzed ("map","mask","MRI")
            
    Outputs:
        coronal_slice : np.ndarray
            Coronal slice obtained
    
    """
    

    
    coronal_slice = img[:,img.shape[1] - 100,:]
        
    if typ == "MRI":
        coronal_slice = gaussian_filter(coronal_slice,0.8)
        
    return np.flipud(coronal_slice)


def getSagittalSlice(img,coronal_com,axial_slice):
    """
    Obtain sagittal slice for collage

    Parameters
    ----------
    img : np.ndarray
        Image to be analyzed
    coronal_com : np.ndarray
        Set of coronal centers of mass
    axial_slice : np.ndarray
        Axial slice coordinates to be included in image

    Returns
    -------
    sagittal_slice : np.ndarray
        Sagittal view for image

    """
    sagittal_slice = img[:,:,img.shape[-1]//2 + 30] # Include part of the leg
    
    sagittal_slice[axial_slice,:] = 0
    sagittal_slice[:,img.shape[1] - 100] = 0
    
    sagittal_slice_aux = binary_erosion(sagittal_slice)
    sagittal_slice_copy = np.copy(sagittal_slice)
        
    return np.fliplr(np.flipud(sagittal_slice_aux*sagittal_slice_copy))
    
    

def getSlices(maps,mri,masks,axial_slices):
    """
    Obtain axial, coronal, and sagittal slices for the collage

    Parameters
    ----------
    maps : list of np.ndarray
        Imiomics maps to be processed
    mri : list of np.ndarray
        Mixed MR images to be processed
    masks : list of np.ndarray
        Bodymasks used in the processing
    axial_slices : list of np.nd
        Axial locations where to do axial slices
        
    Returns
    -------
    axial_maps : list of lists of np.ndarray
        axial slices for the collage (Imiomics maps)
    axial_mri : list of lists of np.ndarray
        axial slices for the collage (mixed MRI)
    axial_masks : list of lists of np.ndarray
        axial slices for the collage (bodymasks)
    coronal_maps : list of np.ndarray
        Coronal slices for the collage (Imiomics maps)
    coronal_mri : list of np.ndarray
        Coronal slices for the collage (mixed MRI)
    coronal_masks : list of np.ndarray
        Coronal slices for the collage (bodymasks)
    sagittal_mri : list of np.ndarray
        Sagittal slices for the collage (mixed MRI)
    sagittal_masks : list of np.ndarray
        Sagittal slices for the collage (bodymasks)

    """
    
    axial_coords = axial_slices*2
    
    axial_maps = []
    axial_mri = []
    axial_masks = []
    coronal_maps = []
    coronal_mri = []
    coronal_masks = []
    sagittal_mri = []
    sagittal_masks = []

    cont = 0

    for (map_img,mri_img,mask,axial_slice) in zip(maps,mri,masks,axial_coords):
        
        mask_aux = mask.copy()
        
        coronal_com,sagittal_com = getCenterofMass(mask) # Obtain center of mass for coronal and sagittal positions
        
        # Coronal slices
        coronal_map_slice = getCoronalSlice(map_img,coronal_com,"map")
        coronal_mri_slice = getCoronalSlice(mri_img,coronal_com,"MRI")
        coronal_mask_slice = getCoronalSlice(mask,coronal_com,"mask")
        #coronal_mask_slice[coronal_mask_slice > 0] = 1

        # sagittal slices (only one for male and for female)
        if cont < len(maps)//2:
            sagittal_mri_slice = getSagittalSlice(np.copy(mri_img),coronal_com,axial_slice)
            sagittal_mask_slice = getSagittalSlice(mask_aux,coronal_com,axial_slice)
            sagittal_mask_slice[sagittal_mask_slice > 0] = 1
            sagittal_mri.append(sagittal_mri_slice)
            sagittal_masks.append(sagittal_mask_slice)
        
        # axial slices
        axial_map_slices = getAxialSlice(map_img,axial_slice)
        axial_mri_slices = getAxialSlice(mri_img,axial_slice)
        axial_mask_slices = getAxialSlice(mask,axial_slice)
        
        axial_maps.append(axial_map_slices)
        axial_mri.append(axial_mri_slices)
        axial_masks.append(axial_mask_slices)
        coronal_maps.append(coronal_map_slice)
        coronal_mri.append(coronal_mri_slice)
        coronal_masks.append(coronal_mask_slice)
        
        cont += 1
        
    return axial_maps,axial_mri,axial_masks,coronal_maps,coronal_mri,coronal_masks,sagittal_mri,sagittal_masks


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def curveLines(maps,masks,mris):
    """
    Get curved lines for plots in the right section of the collage

    Parameters
    ----------
    maps : list of np.ndarray
        Imiomics maps for collage
    masks : list of np.ndarray
        Imiomics masks for collage
    mris : list of lists of np.ndarray
        MR images to know where there is fat tissue and where lean (MRIs are unmixed)

    Returns
    -------
    curves : list of lists
        Extracted curves for later plotting
        [[male_jac],[female_jac],[male_fat],[female_fat]] 
        --> [male_jac] = [negative_adipose,negative_lean,positive_adipose,positive_lean]
    extrema : list of float
        Extreme values for curves, to be used later in collage plots
        [extrema_male_jac,extrema_female_jac,extrema_male_fat,extrema_female_fat] 

    """
    
    curves = [] # List where to store final curves
    extrema = [] # List where to store curves' extrema
    
    for img,mask,mri in zip(maps,masks,mris):
        
        # Indexes where to obtain regions of interest for collage curves
        ind_negative_fat = np.where((img < 0) & (mask > 0) & (mri[0] < mri[1]))
        ind_negative_lean = np.where((img < 0) & (mask > 0) & (mri[0] >= mri[1]))
        ind_positive_fat = np.where((img > 0) & (mask > 0) & (mri[0] < mri[1]))
        ind_positive_lean = np.where((img > 0) & (mask > 0) & (mri[0] >= mri[1]))
        
        inds = [ind_negative_fat,ind_negative_lean,ind_positive_fat,ind_positive_lean] 
        
        curves_map = [] # Store here the curves for the same map
        raw_curves = [] # Store here curves prior to normalization
        norms = [] # Store here values for normalization
        extrema_map = [] # Extreme curve values for the same map
        
        # Compute curves and factors for normalization of each curve
        for enum_ind,ind in enumerate(inds):
            masked_region = np.ones(img.shape)
            masked_region[ind[0],ind[1],ind[2]] = 0

            masked_img = np.ma.array(img,mask=masked_region)
            count = np.ma.sum(np.ma.sum(masked_img,2),1).filled(0)
            mean = np.ma.mean(np.ma.mean(masked_img,2),1).filled(0)
            curve = count*mean # Curve weighted by number of hits and their mean value
            
            curve[:5] = 0 # Clip first 15 slices of curve due to artifacts
            curve[(-5):] = 0 # Clip last 15 slices of curve due to artifacts
            
            if enum_ind%2 == 0: # Find factor to use for curve normalization
                if np.percentile(curve,99) != 0:
                    normalizer = [np.percentile(curve,99)+np.finfo(float).eps]
                else:
                    normalizer = [curve.max() + np.finfo(float).eps]
            else:
                if np.percentile(curve,99) != 0:
                    normalizer.append(np.percentile(curve,99)+np.finfo(float).eps)
                else:
                    normalizer.append(curve.max() + np.finfo(float).eps)
                    
                norms.append(max(normalizer))  
                    
            raw_curves.append(curve)
            
        # Curve normalization
        for enum_curve,curve in enumerate(raw_curves):

            if curve.sum() > 0:
                norm_curve = curve/norms[enum_curve//2] 

            
            norm_curve = moving_average(norm_curve, 7) # Curve filtered with moving average
            
            extrema_map.append(abs(norm_curve.max()))
            extrema_map.append(abs(norm_curve.min()))
                
            if enum_curve < len(raw_curves)//2:
                norm_curve = -norm_curve
            else:
                norm_curve = norm_curve
                
            curves_map.append(norm_curve)
        
        extrema.append(max(extrema_map))
        curves.append(curves_map)
        
    return curves,extrema


def colorbarPlot(cmap,extrema,var,out,tag):
    """
    Get a horizontal colorbar given a colormap and a set of extrema for that colorbar

    Parameters
    ----------
    cmap : np.ndarray
        Colormap to be applied
    extrema : list of int/float
        Set of minima and maxima for the colorbar
    var : str
        Variable whose name needs to be pasted in colorbar
    out : str
        Output folder for colorbar images
    tag : str
        Type of maps that are being analyzed ("beta" for regression, "corr" for correlation, "perc" for percentage increase)

    Returns
    -------
    colorbar_images : list of np.ndarray
        List of np.ndarrays with colorbar images

    """
    
    # Provide custom colormap
    inv_cmap=cmap[::-1]
    color_palette = (inv_cmap/255).tolist()
    
    cmap = LinearSegmentedColormap.from_list('my_list', color_palette, N=256)
    
    if not("logistic" in out.lower()): # Correlations or linear regressions
        if tag == "perc":
            labels = ["%volume increase for {}".format(var),
                      "%fat increase for {}".format(var)]
        else:
            labels = ["Variation in Jacobian volume for {}".format(var),
                      "p.p. increase in FF for {}".format(var)]
    else: # 
        labels = ["Logistic coefficient ({} vs Jacobian Volume)".format(var),
                  "Logistic coefficient ({} vs Fat Fraction)".format(var)]

    
    colorbar_images = [] # List storing colorbar images for the later collage
    
    cont = 0
    
    for lab,extremum in zip(labels,extrema):
    
        if extremum[0]*extremum[1] < 0: # Make a symmetric color scale for the video, leaving the zero point in the middle
            ext = max(np.abs(np.array([extremum[0], extremum[1]])))
            
            if cont == 1 and not("logistic" in out):
                ext = ext/10
                
            minimum = -ext
            maximum = ext
            
        else:
            
            minimum = extremum[0]
            maximum = extremum[1]
            
            if cont == 1 and not("logistic" in out):
                minimum = minimum/10
                maximum = maximum/10
    
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        
        fig = plt.figure(figsize = (5,0.25), dpi = 200)
        ax = plt.gca()
        
        if np.log10(np.abs(maximum)) < -2:
            colorbar = fig.colorbar(cm.ScalarMappable(norm, cmap), ax, orientation = 'horizontal', format='%.0e')
        else:
            colorbar = fig.colorbar(cm.ScalarMappable(norm, cmap), ax, orientation = 'horizontal')
    
        colorbar.set_label(lab)
        
        # Save colorbar image as PNG, load it, and get its correspondent array to be included in the collage
        outname = os.path.join(out,"colorbar_img_{}.png".format(var))
        fig.savefig(outname, bbox_inches = "tight")
        
        colorbar_img = PIL.Image.open(outname)
        colorbar_img = np.asarray(colorbar_img.convert('RGB'))
        colorbar_img = colorbar_img[10:(-10),10:(-10)]
        #os.remove(outname)
        
        colorbar_images.append(colorbar_img)
        
        cont += 1
        
    return colorbar_images



def LUT(array,mri,mask,extrema,cmap):
    """
    Apply Look-Up Table for colormap for a given array

    Parameters
    ----------
    array : np.ndarray
        Imiomics map to be painted
    mri : np.ndarray
        Corresponding MR image to be used as reference background in the collage
    mask : np.ndarray
        Corresponding bodymask to be used in the collage
    extrema : list of float/int
        Set of minima and maxima for scaling the colors of the LUT
    cmap : np.ndarray
        Colormap to be applied

    Returns
    -------
    final_array : np.ndarray
        Final "painted" array with the LUT provided

    """
    
    if extrema[0]*extrema[1] < 0: # Make a symmetric color scale for the video, leaving the zero point in the middle
        ext = max(np.abs(np.array([extrema[0],extrema[1]])))
        minimum = -ext
        maximum = ext
    else:
        minimum = extrema[0]
        maximum = extrema[1]
        
    scale = np.linspace(minimum, maximum, len(cmap))
    cmap = np.flipud(cmap)
    cs = CubicSpline(scale, cmap)
       
    # Convert MRI array to RGB
    mri = np.round((mri - np.amin(mri))*255/(np.amax(mri) - np.amin(mri))).astype(int)
    mri_rgb = np.repeat(mri[:,:,np.newaxis],3,axis=2) 
    final_array = cs(array)
    
    final_array[array > maximum] = np.array([128,0,0]) # Clip potential colors with an intensity larger than the maximum
    final_array[array < minimum] = np.array([0,0,128]) # Clip potential colors with an intensity lower than the maximum
    final_array = np.clip(final_array,0,255).astype(np.uint8)
    final_array[array == 0] = mri_rgb[array == 0] # Place background MRI mix when the map results are not significant
    
    opacity = 0.9 # Set an opacity of 90% for the colored significant sections
    final_array[array != 0] = opacity*final_array[array != 0] + (1-opacity)*mri_rgb[array != 0]
    final_array[mask == 0] = np.array([255,255,255]) # Set background outside the body to white

    
    return final_array


        

def paintSlices(axial_images,coronal_images,sagittal_images,cmap,extrema):
    """
    "Paint" the different axial, coronal, and sagittal slices provided for the Imiomics collage
    with the Look-Up Table provided by the colormap and the extrema

    Parameters
    ----------
    axial_images : list of lists of np.ndarray
        Axial slices, organized as [maps,mri,masks]
    coronal_images : list of lists of np.ndarray
        Coronal slices, organized as [maps,mri,masks]
    sagittal_images : list of lists of np.ndarray
        Sagittal slices, organized as [mri,masks]
    cmap : np.ndarray
        Colormap to be applied to axial and coronal slices
    extrema : list of lists of float
        Set of minima and maxima for the application of the colormap

    Returns
    -------
    mixed_axial : list of lists of np.ndarray
        "Painted" axial slices, organized according to the columns of the Imiomics collage
    mixed_coronal : list of np.ndarray
        "Painted" coronal slices, organized according to the columns of the Imiomics collage
    mixed_sagittal : list of np.ndarray
        "Painted" sagittal slices

    """        
      
    mixed_axial = [] # List for storing mixed axial images
    mixed_coronal = [] # List for storing mixed coronal images
    mixed_sagittal = [] # List for storing mixed sagittal images
    
    # "Paint" the axial and coronal slices
    # Access images columnwise
    for column in range(len(axial_images[0])):
        
        if column < len(axial_images[0])//2:
            extremum = extrema[0]
        else:
            extremum = extrema[1]
        
        # Process axial slices
        axial_column = []
        for i in range(len(axial_images[0][column])):
            axial_img = LUT(axial_images[0][column][i], axial_images[1][column][i], 
                          axial_images[2][column][i], extremum, cmap)

            axial_column.append(axial_img)
            
        mixed_axial.append(axial_column)
            
        # Process coronal slices
        coronal_img = LUT(coronal_images[0][column], coronal_images[1][column], 
                          coronal_images[2][column], extremum, cmap)
        
        # Pad coronal with 10 blank slices on top for collage
        #coronal_img = np.pad(coronal_img,((10,0),(0,0),(0,0)),mode="maximum")
        mixed_coronal.append(coronal_img)
        
    # Process sagittal slices: set to maximum value the areas where the sagittal mask is 0
    for sagittal_ind in range(len(sagittal_images[0])):
        sagittal_img = sagittal_images[0][sagittal_ind].copy()
        sagittal_img[sagittal_images[1][sagittal_ind] == 0] = np.amax(sagittal_img)
        mixed_sagittal.append(sagittal_img[:,40:]) # Remove first 40 empty columns of sagittal plot
        
    return mixed_axial,mixed_coronal,mixed_sagittal


def assembleSlices(coronal_slices,axial_slices):
    """
    Assemble axial slices of the same column into an only array that can be
    plotted into the collage

    Parameters
    ----------
    coronal_slices : list of lists of np.ndarray
        Coronal slices 
    axial_slices : list of lists of np.ndarray
        Axial slices 

    Returns
    -------
    assembled : np.ndarray
        Slices assembled into an only array

    """
    
    cols = [] 
    for i,col in enumerate(axial_slices):
        column = np.concatenate(tuple([coronal_slices[i]] + col),0)
        cols.append(column)
        
    concat = np.concatenate(tuple(cols),1)

    
    s = np.mean(np.mean(concat,2),1)
    ind_remove = np.where(s==255)[0]
    
    concat = np.delete(concat,ind_remove,axis=0)

    
    return concat

    
def figureLayout(axial_imgs,coronal_imgs,sagittal_imgs,colorbar_imgs,curves,curve_extrema,axial_slices,v,n,out):
    """
    Create figure layout for Imiomics collage using Matplotlib GridSpec
    
    Params
    -------
    axial_imgs : list of lists of np.ndarray
        Colored axial slices to be plotted in the final collage
    coronal_imgs : list of np.ndarray
        Colored coronal slices to be plotted in the final collage
    sagittal_imgs : list of np.ndarray
        Sagittal slices to be plotted in the final collage
    colorbar_imgs : list of np.ndarray
        Colorbar images to be plotted in the bottom part of the final collage
    curves : list of lists of np.ndarray
        Set of curves to be plotted in the right-side of the final college
    curve_extrema : list of float
        Extreme values for curves, to be used later in collage plots
    axial_slices : list of np.ndarray
        Set of axial slices where axial images are acquired
    v : list of str
        Variable names for collage
    n : list of str
        Male and female numbers
    out : str
        Folder where to save final output figure
    

    Returns
    -------
    None.

    """
    title_kwargs = dict(ha='center', va='center',fontsize=40,fontweight="bold")
    
    
    fig = plt.figure(constrained_layout=True,figsize=(35,27)) # General figure definition # CHANGE TO (35,28) IF YOU HAVE PROBLEMS WITH THE LAYOUT!!!
    spec = fig.add_gridspec(22,7) # Grid definition fit for collage
    spec.update(hspace=0.001,wspace=0.01) # No vertical spaces between GridSpec subplots
    
    # Title plots
    f_jac_title = fig.add_subplot(spec[0,:2]) # Title for Volume Maps
    f_jac_title.text(0.5,0.5,"Volume",**title_kwargs)
    f_jac_title.set_axis_off()
    f_fat_title = fig.add_subplot(spec[0,2:4]) # Title for FF Maps
    f_fat_title.text(0.5,0.5,"Fat fraction",**title_kwargs)
    f_fat_title.set_axis_off()
    f_slice_title = fig.add_subplot(spec[0,(-3):]) # Title for Sagittal Silhoettes and Curve Plots
    f_slice_title.text(0.5,0.5,"Slice Positions and Slice-Wise Correlations",**title_kwargs)
    f_slice_title.set_axis_off()

    # Image title plots
    gender_titles = ["  Males (N={})  ".format(n[0]),"  Females (N={})  ".format(n[1])]
    gender_titles_aux = gender_titles
    gender_titles_aux = (" ").join(gender_titles_aux)
        
    # Coronal + Axial plots : try to provide a horizontal separation between jacobian and fat fraction
    assemble_jac = assembleSlices(coronal_imgs[:len(coronal_imgs)//2],axial_imgs[:len(axial_imgs)//2]) # Join all axial slices in an array
    assemble_fat = assembleSlices(coronal_imgs[len(coronal_imgs)//2:],axial_imgs[len(axial_imgs)//2:])
    f_assemble_jac = fig.add_subplot(spec[2:(-3),:len(coronal_imgs)//2])
    f_assemble_fat = fig.add_subplot(spec[2:(-3),len(coronal_imgs)//2:len(coronal_imgs)])
    f_assemble_jac.imshow(assemble_jac)
    f_assemble_fat.imshow(assemble_fat)
    f_assemble_jac.set_title(gender_titles_aux,fontsize=30,pad= 50)
    f_assemble_fat.set_title(gender_titles_aux,fontsize=30,pad= 50)
    f_assemble_jac.set_axis_off()
    f_assemble_fat.set_axis_off()

            
    # Colorbar images
    for enum_colorbar,colorbar_img in enumerate(colorbar_imgs):
        f_colorbar = fig.add_subplot(spec[(-3):,(enum_colorbar*2):((enum_colorbar+1)*2)])
        f_colorbar.imshow(colorbar_img)
        f_colorbar.set_axis_off()
        
    # Sagittal images
    sagittal_rows = [2,11]
    for enum_sagittal,sagittal_img in enumerate(sagittal_imgs):
        f_sagittal = fig.add_subplot(spec[sagittal_rows[enum_sagittal]:sagittal_rows[enum_sagittal]+8,-3])
        f_sagittal.imshow(np.roll(sagittal_img,20,1),cmap="gray") # Move sagittal plot to the right
        #f_sagittal.set_ylabel(gender_titles[enum_sagittal].split(" ")[2],fontsize=30,position=(1.0,0.5))
        f_sagittal.text(0,181,gender_titles[enum_sagittal].split(" ")[2],
                        fontsize=30,backgroundcolor="white",
                        horizontalalignment="center",
                        rotation="vertical") # Gender title in right-hand side part of the collage
        f_sagittal.set_title("   Slices",fontsize=35,pad=30)
        
        # Paint axes in white to allow ylabel printing
        f_sagittal.tick_params(axis='x', colors='white')
        f_sagittal.tick_params(axis='y', colors='white')
        f_sagittal.spines['bottom'].set_color('white')
        f_sagittal.spines['top'].set_color('white') 
        f_sagittal.spines['right'].set_color('white')
        f_sagittal.spines['left'].set_color('white')
        #f_sagittal.set_axis_off()
        
    # Curve plots
    cont_curve = 0
    curve_colors = ["lightskyblue","darkblue","orange","darkred"]
    curve_labels = ["Negative adipose","Negative lean","Positive adipose","Positive lean"]
    curve_types = ["Volume","Fat Fraction"]
    organs = ["     heart","     liver","    kidneys","  abdomen","       hip","      thigh "]
    
    for curve_col in range(-2,0):
        for curve_row in sagittal_rows:
            f_curve = fig.add_subplot(spec[curve_row:curve_row+8,curve_col])
            f_curve.set_title(curve_types[cont_curve//2],fontsize=30)
            f_curve.axvline(x=0,ymax=sagittal_img.shape[0],color="black",linewidth=2.0) # Vertical line for symmetry
            
            # Horizontal lines for organ heights and text lines with organ names
            if cont_curve%2 == 0:
                cont_organ = 0
                for axial_slice in axial_slices[0]:
                    f_curve.axhline(axial_slice,xmin=-curve_extrema[cont_curve],
                                    xmax=curve_extrema[cont_curve],linestyle="--",color="black",linewidth=2.0)                   
                    if cont_curve//2 == 0: # Organ names
                        t = f_curve.text(curve_extrema[cont_curve]-0.05,
                                         axial_slice,organs[cont_organ],
                                         fontsize=22)
                        t.set_bbox(dict(edgecolor="white",facecolor="white",alpha=0.5)) # Set transparent text boxes with organ names
                        cont_organ += 1
                    
            else:
                cont_organ = 0
                for axial_slice in axial_slices[1]:
                    f_curve.axhline(axial_slice,xmin=-curve_extrema[cont_curve],
                                    xmax=curve_extrema[cont_curve],linestyle="--",color="black",linewidth=2.0)
                    if cont_curve//2 == 0: # Organ names
                        t = f_curve.text(curve_extrema[cont_curve]-0.05,
                                         axial_slice,organs[cont_organ],
                                         fontsize=22)
                        t.set_bbox(dict(edgecolor="white",facecolor="white",alpha=0.5)) # Set transparent text boxes with organ names
                        cont_organ += 1
            
            # Curve line plots            
            for enum_curve_set,curve_array in enumerate(curves[cont_curve]):
                f_curve.plot(curve_array,np.arange(sagittal_img.shape[0]),curve_colors[enum_curve_set], linewidth=3.0)
                f_curve.xaxis.tick_top()
                f_curve.set_axis_off()
            
            cont_curve += 1
            
    # Curve legend plot: set a dummy plot, get its colors and labels, and plot legend. Then hide everything except the legend
    f_legend = fig.add_subplot(spec[sagittal_rows[-1]-1,(-2):])
    f_legend.set_axis_off()
    for enum_curve_set,curve_array in enumerate(curves[cont_curve-1]):
        f_legend.plot([],[],color = curve_colors[enum_curve_set],label=curve_labels[enum_curve_set],linewidth=3.0)
    f_legend.legend(ncol=2,loc="center",fontsize=25)
    
    # Save final figure
    fig.savefig(os.path.join(out,"{}_collage.png".format(v)))

    
    
def removeColorbarImages(folder):
    """
    Remove colorbar images computed during the obtention of the main collage

    Parameters
    ----------
    folder : str
        Path where colorbar images are located

    Returns
    -------
    Removed colorbar images from desired location

    """

    colorbar_images = sorted(os.listdir(folder))
    for colorbar_image in colorbar_images:
        if "colorbar" in colorbar_image:
            os.remove(os.path.join(folder,colorbar_image))
