import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import median_filter
import sys
import os
import PIL.Image
import cv2
import multiprocessing
from joblib import Parallel, delayed
import math
import copy



def getTitleImage(gender_types, img, diabetes_numbers, title_image, n_men, n_women, r2):
    """
    Get title image for Imiomics collage plot

    Parameters
    ----------
    gender_type : list of str
        Set of genders and types of maps found
    img : np.ndarray
        Map of reference to be used for the obtention of the title image, in diabetes cases
    diabetes_numbers : list of int
        Number of found diabetic patients for each analyzed map (if diabetes is being analyzed)
    title_image : str
        Filename for saving the title image and later loading it
    n_men : int
        Number of analyzed men in the maps
    n_women : int 
        Number of analyzed woment in the maps
    r2 : str
        Flag telling if the title corresponds to a R2 map

    Returns
    -------
    title_img : np.ndarray
        Read title image to be concatenated to the final collage in the top

    """
    fig_text = plt.figure(figsize = (25,1))
    cont = 0

    n_men = str(n_men)
    n_women = str(n_women)
    
    if int(r2) == 1:
        fig_text = plt.figure(figsize = (25,1))
        quotient = 2*len(gender_types)
    else:
        fig_text = plt.figure(figsize = (17,1))
        quotient = 1.1*len(gender_types)
    
    for i,gender_type in enumerate(gender_types):
        if "female" in gender_type:
            if "diabetes" in img.lower():
                if "Jac" in gender_type:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "Vol F (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 Vol F (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
                else:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "FF F (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 FF F (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
            else:
                if "Jac" in gender_type:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "Vol F (N=" + n_women + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 Vol F, (N=" + n_women + ")", fontsize = 15)
                else:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "FF F (N=" + n_women + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 FF F, (N=" + n_women + ")", fontsize = 15)
                        
        else:
            if "diabetes" in img.lower():
                if "Jac" in gender_type:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "Vol M (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 Vol M (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
                else:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "FF M (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 FF M (diab=" + str(diabetes_numbers[cont]) + ")", fontsize = 15)
            else:
                if "Jac" in gender_type:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "Vol M (N=" + n_men + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 Vol M, (N=" + n_men + ")", fontsize = 15)
                else:
                    plt.text(x = (i+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "FF M (N=" + n_men + ")", fontsize = 15)
                    if int(r2) == 1:
                        plt.text(x = (i+len(gender_types)+1)/quotient - 0.03, y = 0.5, ha='right', va='center', s = "R2 FF M, (N=" + n_men + ")", fontsize = 15)
        
        cont += 1
    
    plt.axis('off')
    fig_text.patch.set_visible(False)            
    fig_text.savefig(title_image)
    
    title_img = PIL.Image.open(title_image)
    title_img = np.asarray(title_img.convert('RGB'))
    os.remove(title_image)
    
    return title_img



def customColorbar(cmap, minimum, maximum, label, outname, r2):
    """
    Create custom colorbars for the plots in the animations

    Parameters
    ----------
    ax : Matplotlib axis
        Axis where to do the plotting
    cmap : Matplotlib Colormap
        Desired colormaps for colorbar
    minimum : float
        Minimum values for colorbars
    maxima : list of float
        Maximum values for colorbars
    label : str
        Unit indicator for colorbar
    outname : str
        Filename for colorbar image
    r2 : str
        Flag telling if the colorbar corresponds to a R2 map

    Returns
    -------
    colorbar : List of matplotlib colorbars

    """ 
    
    # Provide custom colormap
    color_palette = (cmap/255).tolist()
    
    cmap = LinearSegmentedColormap.from_list('abc', color_palette, N=256)
    
    if int(r2) != 1:
        if minimum*maximum < 0: # Make a symmetric color scale for the video, leaving the zero point in the middle
            extrema = max(np.abs(np.array([minimum, maximum])))
            minimum = -extrema
            maximum = extrema

    norm = mpl.colors.Normalize(vmin=minimum, vmax=maximum)

    
    fig = plt.figure(figsize = (5,0.25), dpi = 200)
    ax = plt.gca()
    
    if np.log10(np.abs(maximum)) < -2:
        colorbar = fig.colorbar(cm.ScalarMappable(norm, cmap), ax, orientation = 'horizontal', format='%.0e')
    else:
        colorbar = fig.colorbar(cm.ScalarMappable(norm, cmap), ax, orientation = 'horizontal')
        
    colorbar.set_label(label)
    
    fig.savefig(outname, bbox_inches = "tight")
    
    return fig


def arrayColoring(arrays, masks, refs, extrema, colorscale):
    """
    Color arrays with a given color scale, using bodymasks and reference images
    as reference

    Parameters
    ----------
    arrays : list of np.ndarray
        Arrays to be colored
    masks : list of np.ndarrays
        Corresponding bodymasks to be processed
    refs : list of np.ndarrays
        Corresponding reference images to be processed
    extrema : list of float
        Extreme values for plotting
    colorscale : Scipy CubicSpline
        Spline fitting used to color the arrays
        
    Returns
    -------
    final_array : np.ndarray
        Horizontally concatenated and colored arrays
    final_mask : np.ndarray
        Horizontally concatenated corresponding masks

    """
    cont = 0
    
    for array, ref in zip(arrays, refs):
        colored_array = colorscale(array) # Array coloring
        
        colored_array[array > extrema[1]] = np.array([128,0,0]) # Clip upper color
        colored_array[array < extrema[0]] = np.array([0,0,128]) # Clip upper color
        
        colored_array = np.clip(colored_array,0,255).astype(np.uint8) # Array clipping between 0 and 255 and conversion to UINT8
        
        colored_array[array == 0] = ref[array == 0] # Set non-significant regions of the colored array to the reference array
        
        opacity = 0.9
        colored_array[array != 0] = colored_array[array != 0]*opacity + ref[array != 0]*(1-opacity)
        
        mask = masks[cont]
        mask[mask > 0] = 1
        mask = mask.astype("bool")
        colored_array[mask][:] = 255 # Masking the colored array to the bodymask

        
        if cont == 0:
            final_array = np.copy(colored_array)
            final_mask = np.copy(masks[cont])
        else:
            final_array = np.concatenate((final_array, colored_array), axis = 2)
            final_mask = np.concatenate((final_mask, masks[cont]), axis = 2)
        
        cont += 1
    
    return final_array, final_mask
    

def LUT(arrays, masks, refs, r2s, cmap):
    """
    Color given images with a Red-White-Blue colormap (look-up table or LUT)

    Parameters
    ----------
    arrays : list of np.ndarrays
        Arrays to be processed
    masks : list of np.ndarrays
        Corresponding masks to be processed
    refs : list of np.ndarrays
        Corresponding reference images to be processed
    r2s : list of np.ndarrays
        Corresponding r2 maps to be processed (if r2 flag is active)
    cmap : np.ndarray
        Colormap to be applied
        
    Returns
    -------
    final_array : np.ndarray
        Array with concatenated colored arrays
    final_mask : np.ndarray
        Array with concatenated masks
    r2_array : np.ndarray
        Array with horizontally concatenated colored R2 maps. Returned only if r2s is not None
    r2_mask : np.ndarray
        Array with horizontally concatenated bodymasks for R2 maps. Returned only if r2s is not None
    minimum : float
        Minimum value for the colorbar
    maximum : float
        Maximum value for the colorbar

    """
    
    minimum = min([np.amin(array) for array in arrays])
    maximum = max([np.amax(array) for array in arrays])
    
    if r2s is not None:
        maximum_r2 = max([np.amax(r2) for r2 in r2s])
    
    if minimum*maximum < 0: # Make a symmetric color scale for the video, leaving the zero point in the middle
        extrema = max(np.abs(np.array([minimum, maximum])))
        minimum = -extrema
        maximum = extrema

    # Load reference images, normalize them and set them to 4D RGB space (3D + color channels)
    processed_refs = [] # Store reference images in this list
    for ref in refs:
        ref = (255*(ref - np.amin(ref))/(np.amax(ref) - np.amin(ref))).astype(np.uint8) # Normalization
        ref_aux = np.repeat(ref,3,-1) # Build RGB space
        ref = ref_aux.reshape((ref.shape[0], ref.shape[1], ref.shape[2], 3)) # Reshape to 4D space
        processed_refs.append(ref)
    
    scale = np.linspace(minimum, maximum, len(cmap)) # Coloring scale to be applied

    cs = CubicSpline(scale, cmap) # LUTing
    
    final_array, final_mask = arrayColoring(arrays, masks, processed_refs, [minimum,maximum], cs) # LUT application to arrays
    
    if r2s is not None:
        if maximum_r2 < 0.3:
            scale = np.linspace(0, 0.3, len(cmap)) # R2 maps are left with a scale from 0 to 0.3, so to be comparable to other computations 
        else:
            scale = np.linspace(0, maximum_r2, len(cmap)) # If the R2 map has high scores, expand its contrast until the maximum score

        cs_r2 = CubicSpline(scale, cmap)
        r2_array, r2_mask = arrayColoring(r2s, masks, processed_refs, [0,maximum_r2], cs_r2) # LUT application to R2 arrays
  
        return final_array, final_mask, r2_array, r2_mask, maximum_r2, minimum, maximum
    
    else:
        return final_array, final_mask, minimum, maximum


def referenceImages(mri):
    """
    Prepare reference images

    Parameters
    ----------
    mri : list of str
        List with W and F image paths for template

    Returns
    -------
    ref : np.array 
        Combination of W and F for male template

    """
    if len(mri) != 2:
        print("Wrong number of reference images given for the same ID")
        sys.exit(2)
    else:
        # Load images with SimpleITK + numpy. Set FF image as initial reference
        for mri_image in mri:
            if not os.path.exists(mri_image):
                print("Reference image {} does not exist. Please provide existing reference images".format(mri_image))
                sys.exit(2)
            else:
                try:
                    image = np.flip(sitk.GetArrayFromImage(sitk.ReadImage(mri_image)),1)
                    if "W" in mri_image:
                        #ref = np.copy(image)
                        w = np.clip(image,np.percentile(image.flatten(),1),np.percentile(image.flatten(),99))
                    elif "F" in mri_image:
                        #w = np.copy(image)
                        f = np.clip(image,np.percentile(image.flatten(),1),np.percentile(image.flatten(),99))
                except:
                    print("Reference image {} could not be read".format(mri_image))
                    sys.exit(2)
        
        # In those places where the WF is low, substitute by the FF
        ref = 0.5*(w + f)
        #ref[ref < np.median(ref.flatten())] = 0.25*w[ref < np.median(ref.flatten())]
        
        return ref
    

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


def mapLoader(path, mask_paths, method, r2):
    """
    Load Imiomics map in given path. If R2 maps are required, they are also plotted

    Parameters
    ----------
    path : str
        Path with Imiomics map
    mask_paths : str
        Set of mask file or files
    method : str or None
        Multiple Comparison Correction method (None, or "holm","hochberg","rft")

    Returns
    -------
    map_image : np.ndarray
        Loaded map image
    mask_image : np.ndarray
        Loaded mask image
    r2_image : np.ndarray
        Loaded R2 map. Returned only if r2 flag is 1
    p1 : float
        Percentile 1 of map, for later statistics
    p1_r2 : float
        Percentile 1 of R2 map, for later statistics, returned only if r2 flag is 1
    p99 : float
        Percentile 99 of map, for later statistics
    p1_r2 : float
        Percentile 99 of R2 map, for later statistics, returned only if r2 flag is 1
    gender : str
        Gender of map that is analyzed
    typ : str
        Type of map analyzed: "Fat" for FF, "Jac" for Jacobian
    N : list of int
        Number of subjects analyzed per gender

    """
    
    if not os.path.exists(path):
        print("File '{}' does not exist. Please provide an existing file".format(path))
        sys.exit(2)
    else:
        diabetes_cont = 0 # Subject counter for diabetes maps
        try: 
            map_image = np.flip(sitk.GetArrayFromImage(sitk.ReadImage(path)),1)
        except:
            print("Image file '{}' could not be opened".format(path))
            sys.exit(2)
            
        # Determine gender of Imiomics map
        file = path.split("/")[-1]
        file = file.split(".")[0]
        
        if "/male" in path.lower():
            gender = "male"
            N = int(file.split("_")[-1]) 
            if "diabetes" in path.lower():
                #diabetes_cont = 1200
                diabetes_cont = 1083
            
        elif "/female" in path.lower():
            gender = "female"
            N = int(file.split("_")[-1])
            if "diabetes" in path.lower():
                #diabetes_cont = 528
                diabetes_cont = 499
                
        else:
            print("Map gender not found. Please provide maps with a valid gender ('Male'/'Female')")
            sys.exit(2)
            
        # Determine type of Imiomics map
        if "jac" in path.lower():
            typ = "Jac"
        elif "fat" in path.lower():
            typ = "Fat"
        else:
            print("Map type not found. Please provide maps with a valid type ('Jac'/'Fat')")
            
        # Load bodymask
        if "," in mask_paths:
            mask_files = mask_paths.split(",")
        else:
            mask_files = [mask_paths]
            
        # Find corresponding mask file for the desired gender
        ind_mask = [i for i, s in enumerate(mask_files) if '/' + gender.lower() in s]
                
        mask_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_files[ind_mask[0]]))
        mask_image = np.flip(mask_image,1)
            
        # Look if there are p value maps in the same folder as the original maps. 
        # If so, use them for filtering for significant information
        t1 = "beta" in path and os.path.exists(path.replace("beta", "p"))
        t2 = "corr" in path and os.path.exists(path.replace("corr", "p"))
        t3 = "mean" in path and os.path.exists(path.replace("mean", "p"))
        t4 = "std" in path and os.path.exists(path.replace("std", "p"))
        
        if t1 or t2 or t3 or t4:
            
            if t1:
                p_file = path.replace("beta", "p")
            elif t2:
                p_file = path.replace("corr", "p")
            elif t3:
                p_file = path.replace("mean", "p")
            elif t4:
                p_file = path.replace("std", "p")
            
            if int(r2) == 1:
                file = path.split("/")[-1]
                if os.path.exists(path.replace("beta" + file[4], "rsquared")):
                    r2_file = path.replace("beta" + file[4], "rsquared")
                elif os.path.exists(path.replace("corr" + file[4], "rsquared")):
                    r2_file = path.replace("corr" + file[4], "rsquared")
                    
            
            # Existing p value maps
            significance = (0.05/18) # significance = 0.05
            try:
                p_image = np.flip(sitk.GetArrayFromImage(sitk.ReadImage(p_file)),1)
                p_mask = multipleComparison_correction(p_image,mask_image,method,significance)
                #p_mask = np.copy(mask_image)
                
                if "longitudinal" in p_file.lower(): # Remove noisy voxels from longitudinal maps
                    filtered_mask = median_filter(p_mask,size=3)
                    p_mask *= filtered_mask
                
                map_image *= p_mask
                if int(r2) == 1:

                    r2_image = np.flip(sitk.GetArrayFromImage(sitk.ReadImage(r2_file)),1)
                    r2_image *= p_mask
                    
                    # Remove NaN values from R2 image (set them to 0)
                    nan = np.where(np.isnan(r2_image))
                    r2_image[nan] = 0
    
            except:
                print("p value map file '{}' could not be opened or map and p value map do not have same size".format(p_file))
                sys.exit(2)
                
        # Clip map between percentiles 1 and 99, for outlier removal
        p1 = np.percentile(map_image[map_image != 0].flatten(), 1)
        p99 = np.percentile(map_image[map_image != 0].flatten(), 99)
        p1_min = np.amin(map_image[map_image != 0])
        p99_max = np.amax(map_image[map_image != 0])
        
        if ((p1 > 0) and (p1_min < 0)) or ((p1 == p99) and (p1==0.0)):
            p1 = p1_min
        if ((p99 < 0) and (p99_max > 0)) or ((p1 == p99) and (p1==0.0)):
            p99 = p99_max   
        
        if p99 == 0:
            map_image[map_image < p1_min] = p1_min
            map_image[map_image > p99_max] = p99_max
        else:
            map_image[map_image < p1] = p1
            map_image[map_image > p99] = p99
            
        
        if int(r2) == 1:
            p1_r2 = np.amin(r2_image)
            p99_r2 = np.amax(r2_image)
            r2_image[r2_image < p1_r2] = p1_r2
            r2_image[r2_image > p99_r2] = p99_r2
        
        
        
        if int(r2) == 1:
            return map_image,mask_image,r2_image, gender, typ, N, diabetes_cont
        else:
            return map_image,mask_image,gender, typ, N, diabetes_cont
    
    
def refImageLoader(paths):
    """
    Load reference files from a list of paths

    Parameters
    ----------
    paths : list of str
        Paths with reference files

    Returns
    -------
    ref_images : list of np.array
        Reference image arrays

    """
    
    path_ids = []
    paths_array = np.array(paths)
    
    for path in paths:
        final_path = path.split("/")[-1]
        path_ids.append(int(final_path.split("_")[0]))
        
    path_ids = np.array(path_ids).astype(int)
    _,unique_ind = np.unique(path_ids, return_index=True)
    unique_ids = path_ids[np.sort(unique_ind)]
    ref_images = [] # List where to store the final images
    
    for unique_id in unique_ids:
        ind = np.where(path_ids == unique_id)[0]
        mri_paths = paths_array[ind].tolist()
        ref_images.append(referenceImages(mri_paths))
    
    return ref_images


def maskLoader(paths, genders):
    """
    Load masks from given paths and genders
    
    Parameters
    ----------
    paths : list of str
        Filepaths of bodymask files
    genders : list of str
        Genders to analyze

    Returns
    -------
    masks : list of np.array
        Loaded masks for genders

    """
    
    masks = [None]*len(genders)
    genders_array = np.array(genders)
    
    for path in paths:
        if not os.path.exists(path):
            print("Bodymask file {} does not exist. Please provide an existing file".format(path))
            sys.exit(2)
        else:
            try:
                mask = np.flip(sitk.GetArrayFromImage(sitk.ReadImage(path)),1).astype(int)
                mask[mask > 0] = 1
            except:
                print("Bodymask file {} could not be read".format(path))
                
            if "female" in path.lower():
                try:
                    ind = np.where(genders_array == "female")[0]
                except:
                    print("Female masks not found in list of masks. Please provide valid bodymask files")
                    sys.exit(2)
            else:
                try:
                    ind = np.where(genders_array == "male")[0]
                except:
                    print("Male masks not found in list of masks. Please provide valid bodymask files")
                    sys.exit(2)
                
            for i in ind:
                masks[i] = mask
                
    return masks
    
def parallelVideoWriter(final_array, final_mask, colorbar_resized_images, new_rows, title_image, title_rows, var, out, key):
    """
    Write video given joint arrays, masks, map types, colorbar images,
    destination folder and orientation. This is done in parallel for 
    axial and coronal orientations

    Parameters
    ----------
    final_array : np.array
        Joint array of colormaps + reference images
    final_mask : np.array
        Joint bodymasks
    colorbar_images : list of np.ndarray
        Colorbar images to introduce in the video
    colorbar_rows : int
        Rows ocuppied in the final video file by the colorbar images
    title_image : np.ndarray
        Read title image to be concatenated to the final videos in the top
    title_rows : int
        Rows occupied in the top of the video by the title
    var : str
        Variable name to be saved
    out : str
        Output path where to save video file
    key : str, optional
        Video orientation. The default is "coronal".

    Returns
    -------
    Saved video file in output folder

    """

    writeVideo(final_array, final_mask, colorbar_resized_images, new_rows, title_image, title_rows, var, out, key)  
    

def writeVideo(final_array, final_mask, colorbar_images, colorbar_rows, title_image, title_rows, var, out, key = "coronal"):
    """
    Write video given joint arrays, masks, map types, colorbar images,
    destination folder and orientation

    Parameters
    ----------
    final_array : np.array
        Joint array of colormaps + reference images
    final_mask : np.array
        Joint bodymasks
    colorbar_images : list of np.ndarray
        Colorbar images to introduce in the video
    colorbar_rows : int
        Rows ocuppied in the final video file by the colorbar images
    title_image : np.ndarray
        Read title image to be concatenated to the final videos in the top
    title_rows : int
        Rows occupied in the top of the video by the title
    var : str
        Variable name to be saved
    out : str
        Output path where to save video file
    key : TYPE, optional
        Video orientation. The default is "coronal".

    Returns
    -------
    Saved video file in output folder

    """
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    if key.lower() == "axial": # Axial video
        
        # State videowriter
        writer = cv2.VideoWriter(os.path.join(out, var + '_axial.avi'), fourcc, 20, (final_array.shape[2],final_array.shape[1] + colorbar_rows + title_rows))
        
        # Provide videoframes
        
        for i in range(final_array.shape[0]):
            s = np.flipud(final_array[i]) # Upside-down flipping map along Z
            aux = np.copy(s)
            aux[np.flipud(final_mask[i]) == 0] = np.array([255,255,255]) # Set external part of image to white
            s = aux.astype(np.uint8)
            final_slice = np.zeros((s.shape[0] + colorbar_images[0].shape[0], s.shape[1], s.shape[-1])) + 255 # Final slice
            final_slice[:s.shape[0],:,:] = s
            row_count = 0
            for colorbar_image in colorbar_images:  # Insert colorbar in each videoframe
                final_slice[s.shape[0]:,row_count:(row_count + final_slice.shape[1]//len(colorbar_images)),:] = colorbar_image*255
                row_count += final_slice.shape[1]//len(colorbar_images)

            # Add title at the top
            final_slice = np.concatenate((title_image*255, final_slice), 0)

            # Processed videoframe
            final_slice = final_slice.astype(np.uint8)
            
            # Write videoframe
            writer.write(final_slice)
            
    elif key.lower() == "coronal": # Coronal video
        
        # State videowriter
        writer = cv2.VideoWriter(os.path.join(out, var + '_coronal.avi'), fourcc, 20, (final_array.shape[2],final_array.shape[0] + colorbar_rows + title_rows))
        
        # Provide videoframes
        for i in range(final_array.shape[1]):
            s = np.flip(final_array[:,i,:,:],0)
            aux = np.copy(s)
            aux[np.flip(final_mask[:,i,:],0) == 0] = np.array([255,255,255])
            s = aux.astype(np.uint8)
            final_slice = np.zeros((s.shape[0] + colorbar_images[0].shape[0], s.shape[1], s.shape[-1])) + 255
            final_slice[:s.shape[0],:,:] = s
            row_count = 0
            for colorbar_image in colorbar_images:  # Insert colorbar in each videoframe
                final_slice[s.shape[0]:,row_count:(row_count + final_slice.shape[1]//len(colorbar_images)),:] = colorbar_image*255
                row_count += final_slice.shape[1]//len(colorbar_images)
            
            # Add title at the top
            final_slice = np.concatenate((title_image*255, final_slice), 0)
            
            # Processed videoframe
            final_slice = final_slice.astype(np.uint8)
            
            # Write videoframe
            writer.write(final_slice)

    
    else:
        print("Wrong video orientation provided. Please provide 'axial' or 'coronal' as possible video frames")
        sys.exit(2)
