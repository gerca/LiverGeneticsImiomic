import getopt
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
import cmapy
from matplotlib import cm
from scipy.interpolate import CubicSpline
from matplotlib.colors import LinearSegmentedColormap
import PIL.Image
import skimage.transform
import sys
import time
import multiprocessing
from joblib import Parallel, delayed
import videoHelper


def main(argv):
    """
    Main function. Plot axial and coronal videos from Imiomics maps
    @param argv: System arguments

    - m : Imiomics maps to be represented in the videos. If they are regression maps, they are usually the maps
          starting with "beta1" tags. They include the male-Jac maps (with "Jac_M" tag), the female-Jac maps
          (with "Jac_F" tag), the male-Fat maps (with "Fat_M" tag), and the female-Fat maps (with "Fat_F" tag).
          The map NRRD files should be inputted in that order, separated by commas

    - b : VTK bodymask files to show only the body voxels of the imiomics maps in the videos. They are usually
          located in the "collages_data/template_name/gender" folder (where "template_name" is the name of the 
          registration template used for the computation of the Imiomics maps to be represented in the videos,
          while "gender" is "female" or "male"). They have to be inputted as male-bodymask-map and female-
          bodymask-map, separated by commas

    - r : NRRD reference files consisting of water and fat MRI of the template subject used in the registration process
          to get the Imiomics maps. They are usually located in the "collages_data/template_name/gender" folder (where 
          "template_name" is the name of the registration template used for the computation of the Imiomics maps to be 
          represented in the videos, while "gender" is "female" or "male"). These files have to be inputted as
          male-water-MRI, male-fat-MRI, female-water-MRI, female-fat-MRI, separated by commas

    - v : variable name of metadata used in Imiomics maps computation

    - f : whether to plot R2 maps (with a 1) or not (with a 0)

    - c : multiple comparison correction (MCC) method to be applied (optional). As statistical tests are computed
          voxelwise, there are multiple executions of statistical tests. Thus, there can be errors induced by the 
          multiple comparisons. There are different MCC methods that can be applied: "holm" for Holm s 
          multiple correction method, "hochberg" for Hochberg s method, and "rft" for Random Field Theory

    - o : Output folder for axial and coronal videos


    Many helper functions are used by this script, in "videohelper.py"


    @return: void
    """
    init = time.time()
    
    maps = ""
    bodymasks = ""
    ref = ""
    var = ""
    out = ""
    mcc = None # Multiple Comparison Correction method
    r2 = "0"
    
    # Read Input Parameters
    try:
        opts, args = getopt.getopt(argv, "hm:b:r:v:f:c:o:", ["maps=","bodymasks=","variable=","r2_flag=","mcc=","output="])
    except getopt.GetoptError:
        print("imiomicsVideos.py -m <map_files> -b <bodymask_files> -r <reference_files> -v <variable> -f <r2_flag> -c <mcc> -o <output_folder>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("imiomicsVideos.py -m <map_files> -b <bodymask_files> -r <reference_files> -v <variable> -f <r2_flag> -c <mcc> -o <output_folder>")
            sys.exit()
        elif opt in ("-m", "--maps"):
            maps = arg
        elif opt in ("-b", "--bodymasks"):
            bodymasks = arg
        elif opt in ("-r", "--reference"):
            ref = arg
        elif opt in ("-v", "--variables"):
            var = arg
        elif opt in ("-f", "--r2_flag"):
            r2 = arg
        elif opt in ("-c", "--mcc"):
            mcc = arg
        elif opt in ("-o", "--output"):
            out = arg
            
    # Handle user input
    n_cpus = multiprocessing.cpu_count() 

    
    # Load Imiomics map and bodymasks with SimpleITK, and filter for p values if information is available
    if maps == "":
        print("Unavailable Imiomics maps. Please provide Imiomics maps files")
        sys.exit(2)
    else:
        if "," in maps: # Several map files provided
            maps = maps.split(",")
            map_images = [] # Loaded map images
            masks = [] # Loaded mask images
            maxima = [] # Map maxima
            minima = [] # Map minima
            genders = [] # Genders that are analyzed
            types = [] # Map types that are analyzed
            N = [] # Number of subjects from each map
            diabetes_cont = [] # Number of diabetic subjects per map (in case diabetes is assessed)
            n_men = 0
            n_women = 0
            if int(r2) == 1:
                r2_images = [] 
                maxima_r2 = []
                minima_r2 = []

            for m in maps:
                if int(r2) == 1:
                    map_image, mask_image, r2_image, gender, typ, n, diab_cont = videoHelper.mapLoader(m,bodymasks,mcc,r2)
                    r2_images.append(r2_image)
                else:
                    map_image, mask_image, gender, typ, n, diab_cont = videoHelper.mapLoader(m,bodymasks,mcc,r2)
                    
                map_images.append(map_image)
                masks.append(mask_image)
                genders.append(gender)
                types.append(typ)
                N.append(n)
                diabetes_cont.append(diab_cont)
                
            # Get number of analyzed men and women
            for gender, n in zip(genders, N):
                if gender == "male":
                    n_men = n
                elif gender == "female":
                    n_women = n

                    
        else: # Only one map given for video processing
            
            if int(r2) == 1:
                map_image, mask_image, r2_image, gender, typ, n, diab_cont = videoHelper.mapLoader(maps,bodymasks,mcc,r2)
                r2_images = [r2_image]
            else:
                map_image, mask_image, gender, typ, n, diab_cont = videoHelper.mapLoader(maps,bodymasks,mcc,r2)
                
            map_images = [map_image]
            masks = [mask_image]
            genders = [gender]
            types = [typ]
            N = [n]
            diabetes_cont = [diab_cont]
            maps = [maps]
            
            n_men = 0
            n_women = 0
            # Get number of analyzed men and women
            if gender == "male":
                n_men = n
            elif gender == "female":
                n_women = n


    gender_types = [gender + "_" + typ for gender, typ in zip(genders, types)] # Combine genders and types into an only list   
        
    # Load reference images   
    if not "," in ref:
        print("Only one reference image provided. Need both W and F images")
        sys.exit(2)
    else:
        ref = ref.split(",")
        ref_images = videoHelper.refImageLoader(ref)
        if 2*len(ref_images) == len(map_images): # Repeat reference images for analysis of Jac and FF
            ref_images *= 2
    
        
    # Examine output folder given
    if out == "":
        print("Output folder not given. Outputting results to current directory")
        out = os.getcwd() 
    elif not(os.path.exists(out)):
        print("Output folder does not exist. Creating it...")
        os.makedirs(out)
        
    # Examine variable name given
    if var == "":
        print("No name given to map variable. Working with empty name...")

    # Process inputted data
    types_array = np.array(types)
    genders_array = np.array(genders)
    
    # Colormap: jet (GBR for openCV library)
    colormaps = np.asarray([[128, 0, 0],  # Deep blue

                               [255, 0, 0],  # Blue

                               [255, 255, 0],  # Cyan

                               [255, 255, 255],  # White

                               [0, 165, 255],  # Orange

                               [0, 0, 255],  # Red

                               [0, 0, 128]])  # Deep red

    colorbar_images = [] # List storing colorbar images used in videos
    r2_colorbar_images = [] # List storing colorbar images for R2 maps used in videos
     
    # Image coloring according to inner colormap
    unique_types = np.flip(np.unique(types_array))
    unique_genders = np.flip(np.unique(genders_array))
    
    for q,unique_type in enumerate(unique_types): # iterate through the types of imiomics maps given
        maps_to_process = [] # List of maps to process
        masks_to_process = [] # List of masks to process
        refs_to_process = [] # List of refs to process
        if int(r2) == 1:
            r2_to_process = [] # List of R2 maps to process 
        for unique_gender in unique_genders:
            ind = np.where((types_array == unique_type) & (genders_array == unique_gender))[0]
            maps_to_process.append(map_images[ind[0]])
            masks_to_process.append(masks[ind[0]])
            refs_to_process.append(ref_images[ind[0]])
            if int(r2) == 1:
                r2_to_process.append(r2_images[ind[0]])
            
            
        # Color images
        if int(r2) != 1:
            color_array, color_mask, minimum, maximum = videoHelper.LUT(maps_to_process, masks_to_process, refs_to_process, None, colormaps)
        else:
            color_array, color_mask, color_r2, mask_r2, maximum_r2, minimum, maximum = videoHelper.LUT(maps_to_process, masks_to_process, refs_to_process, r2_to_process, colormaps)

        
        if q == 0:
            final_array = np.copy(color_array)
            final_mask = np.copy(color_mask)
            if int(r2) == 1:
                final_r2_array = np.copy(color_r2)
                final_r2_mask = np.copy(mask_r2)
        else:
            final_array = np.concatenate((final_array, color_array), 2)
            final_mask = np.concatenate((final_mask, color_mask), 2)
            if int(r2) == 1:
                final_r2_array = np.concatenate((final_r2_array, color_r2), 2)
                final_r2_mask = np.concatenate((final_r2_mask, mask_r2), 2) 
            
        # Colorbar processing
        if unique_type == "Jac":
            label = "Increase in Jacobian Volume for " + var + ", M=" + str(n_men) + "/F=" + str(n_women)
            
            if "logistic" in maps[0].lower():
                label = "Logistic coefficient ({} vs Volume), M={}, F={}".format(var,n_men,n_women)
            
            colorbar_file = os.path.join(out, "jac_colorbar_" + var + ".png")
            if int(r2) == 1:
               label_r2 = "r2-fit for Jacobian Volume for " + var + ", M=" + str(n_men) + "/F=" + str(n_women) 
               r2_colorbar_file = os.path.join(out, "jac_colorbar_r2_" + var + ".png")
        elif unique_type == "Fat":
            label = "Increase in FF for " + var + ", M=" + str(n_men) +"/F=" + str(n_women)
            
            if "logistic" in maps[0].lower():
                label = "Logistic coefficient ({} vs Fat Fraction), M={}, F={}".format(var,n_men,n_women)
            
            colorbar_file = os.path.join(out, "ff_colorbar_" + var + ".png")
            minimum = minimum/10
            maximum = maximum/10
            if int(r2) == 1:
               label_r2 = "r2-fit for FF for " + var + ", M=" + str(n_men) +"/F=" + str(n_women) 
               r2_colorbar_file = os.path.join(out, "ff_colorbar_r2_" + var + ".png")


        # Produce colorbar and save it in output folder
        colorbar = videoHelper.customColorbar(colormaps, -2, 2, label, colorbar_file, 0)
        colorbar_img = PIL.Image.open(colorbar_file)
        colorbar_img = np.asarray(colorbar_img.convert('RGB'))
        os.remove(colorbar_file)
        colorbar_images.append(colorbar_img)

        if int(r2) == 1: # Produce colorbars for the R2 maps
            if maximum_r2 < 0.3:
                r2_colorbar = videoHelper.customColorbar(colormaps, 0, 0.3, label_r2, r2_colorbar_file, 1)
            else:
                r2_colorbar = videoHelper.customColorbar(colormaps, 0, maximum_r2, label_r2, r2_colorbar_file, 1)
            
            r2_colorbar_img = PIL.Image.open(r2_colorbar_file)
            r2_colorbar_img = np.asarray(r2_colorbar_img.convert('RGB'))
            os.remove(r2_colorbar_file)
            r2_colorbar_images.append(r2_colorbar_img)

    
    # If R2 maps are given, concatenate them horizontally to the other maps
    if int(r2) == 1:    
        final_array = np.concatenate((final_array, final_r2_array), 2)
        final_mask = np.concatenate((final_mask, final_r2_mask), 2)
        factor = 10
    else:
        factor = 7

    # Set number of rows of final colorbar image
    new_rows = int(4*np.ceil((final_array.shape[1]*colorbar_img.shape[0])/(len(np.unique(types_array))*colorbar_img.shape[1])))
    

    # Get title image to set in the top part of the videos
    title_image = os.path.join(out, "titles.png")
    title_image = videoHelper.getTitleImage(gender_types, maps[0], diabetes_cont, title_image, n_men, n_women, r2)
    title_image = title_image[:,title_image.shape[1]//9:(-title_image.shape[1]//9)]
    
    
    # Resize title image to fit in video
    title_rows = int(factor*np.ceil(final_array.shape[1]*title_image.shape[0])/(title_image.shape[1]))
    title_image = skimage.transform.resize(title_image, (title_rows, final_array.shape[2], title_image.shape[-1]))
    
    # Resize the given colorbar images to fit in the videos
    colorbar_resized_images = []
    for colorbar_image in colorbar_images:
        colorbar_img = skimage.transform.resize(colorbar_image, (new_rows, final_array.shape[2]//(len(colorbar_images) + len(r2_colorbar_images)), colorbar_image.shape[-1]))
        colorbar_resized_images.append(colorbar_img)
    
    # Append the R2 colorbar images, if available
    if int(r2) == 1:
        for r2_colorbar_image in r2_colorbar_images:
            colorbar_img = skimage.transform.resize(r2_colorbar_image, (new_rows, final_array.shape[2]//(len(colorbar_images) + len(r2_colorbar_images)), colorbar_image.shape[-1]))
            colorbar_resized_images.append(colorbar_img)
            
    keys = ["axial", "coronal"] # Compute axial and coronal videos in parallel
    for i in range(2):
        Parallel(n_cpus)(delayed(videoHelper.parallelVideoWriter)(final_array, final_mask, colorbar_resized_images, new_rows, title_image, title_rows, var, out, key) for key in keys)
        
    print("Done! Ellapsed time: {} seconds".format(round(time.time() - init, 2)))



if __name__ == '__main__':
    main(sys.argv[1:])        
