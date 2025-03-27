import h5py
import numpy as np
from scipy import stats
import SimpleITK as sitk
from joblib import Parallel, delayed
import multiprocessing
import time
import sys
import getopt
import os
import statsmodels.api as sm
import itertools
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def main(argv):
    """
    Main function
    @param argv: System arguments
    @return: void
    """
    fn = ''  # file name
    grp_n = ''  # group name
    var_f = ''  # variable file (age file)
    out = ''  # vtk file constructed
    ids_f = ''  # file that has a list of specific ids to use in the correlation calculation
    ids = []  # list of ids to use in case there is a ids file
    var = []  # list of values from the variable we want to use from the metadata (age)
    compute = '' # Flag for computational mode
    mask = '' # path to mask
    
    # Perform multiple linear or logistic regression cross-sectional voxel-wise maps, using TXT files with information on the main variable of interest, and covariates
    # All the maps for the coefficients and p-values of the main variable of interest and the covariates are computed in separate NRRD files, together with a R2-fit map
    # Result order is 0 for intercept, 1 for variable of interest, and 2,3...N for covariates, in the order they are given in the shell file

    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore', ConvergenceWarning)

    # Read Input Parameters
    try:
        opts, args = getopt.getopt(argv, "hf:g:v:i:l:m:c:o:r:", ["file=", "group=", "variable=", "ids=", "labels=", "mask=", "compute=", "output="])
    except getopt.GetoptError:
        print(
            "regression.py -f <path_to_hdf5_file> -g <group_name> -v <variable(s).txt> -i <ids.txt> -l <main_variable.txt> -m <mask__path_vtk> -c <computation_mode> -o <output__name_vtk>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("regression.py -f <path_to_hdf5_file> -g <group_name> -v <variable(s).txt> -i <ids.txt> -l <main_variable.txt> -m <mask__path_vtk> -c <computation_mode> -o <output__name_vtk>")
            sys.exit()
        elif opt in ("-f", "--file"):
            fn = arg
        elif opt in ("-g", "--group"):
            grp_n = arg
        elif opt in ("-v", "--variable"):
            var_f = arg
        elif opt in ("-i", "--ids"):
            ids_f = arg
        elif opt in ("-l", "--labels"):
            labels_f = arg
        elif opt in ("-m", "--mask"):
            mask = arg
        elif opt in ("-c", "--compute"):
            compute = arg
        elif opt in ("-o", "--output"):
            out = arg

    # Validate that the required parameters have been given
    if fn == '' or grp_n == '' or out == '' or var_f == '':
        print("regression.py -f <path_to_hdf5_file> -g <group_name> -v <variable(s).txt> -i <ids.txt> -l <main_variable.txt> -m <mask__path_vtk> -o <output__name_vtk>")
        sys.exit(2)
        
    # If there is a file with ids retrieve the ids
    if ids_f != '':
        ids_r = open(ids_f, 'r')  # Open file with the IDs we want to use for computation
        ids = [line.strip() for line in ids_r.readlines()]  # load all the values
        ids_r.close()  # close the text file
        
    # If no labels are given, assume everything as zero
    if labels_f == "":
        labels = np.zeros(len(ids))
    else:
        if compute.lower() == "multiple":
            labels = np.loadtxt(labels_f)
        elif compute.lower() == "logistic":
            labels = np.loadtxt(labels_f).astype(int)
        else:
            print("Wrong computation mode. Please input 'multiple' for multiple regression or 'logistic' for logistic regression")
            sys.exit(2)
            
        if len(labels) != len(ids):
            print("Different number of labels and IDs given. Please input the same number")
            sys.exit(2)
            

    # The variable section is made up of several variables separated by commas. Extract all of them 
    var_f = var_f.split(",") # The first variable is the main one, the rest are covariates
    var = [] # List where the multiple variables will be stored
    for i,var_list in enumerate(var_f):
        var_r = open(var_list, 'r') # Open file with the variable we want to use for computation
        
        if not("ids" in var_f) and not("IDs" in var_f):
            v = np.array([float(line.strip()) for line in var_r.readlines()])  # load all the values
        else:
            v = [line.strip() for line in var_r.readlines()]  # load all the values
        
        if not "," in ids_f:
            if i == 0: 
                v_aux = v
            else:
                if len(v_aux) != len(v):
                    print("The files introduced have a different number of values. Please introduce files with the same number of values")
                    sys.exit(2)
        var_r.close()  # close the text file
        var.append(v)
        v_aux = v
        
    # Insert the labels in the variable list, and set the labels as the first variable
    if compute == "multiple":
        var = [labels] + var
    
    # Compute the number of images to be outputted: 2*(#covariates + 2)
    num = 2*(len(var_f) + 2)
    parent_folder = "/" + "/".join(out.split("/")[:(-1)]) + "/"
    out_filenames = []
    for i in range(num):
        if i < num//2:
            out_filenames.append(parent_folder + "beta" + str(i % (num//2)) + out.split("/")[-1]) 
        else:
            out_filenames.append(parent_folder + "p" + str(i % (num//2)) + out.split("/")[-1])
            
    out_filenames.append(parent_folder + "rsquared" + out.split("/")[-1])
    num += 1
    
    # If there is a mask, load it with SimpleITK
    if mask != '':
        coord = maskLoader(mask)

    print("Initializing...")
    print("Params...")
    print("Data:", fn)
    print("Group:", grp_n)
    print("Variable(s):", var_f)
    print("Ids:", ids_f)
    print("Computation mode: {} regression".format(compute))
    print("Labels:", labels_f)
    print("Mask:", mask)
    print("Output:", out)

    num_cores = multiprocessing.cpu_count()  # Get the number of cores available
    init = time.time()  # Obtain the current time
    

    f = h5py.File(fn, 'r')  # Open the hdf5 file in read mode.
    grp = f.get(grp_n)  # Load a group of datasets - volume or fat
    
    keys = ids if not len(ids) == 0 else grp.keys()  # Assign the ids if given or all keys = all ids
    count = len(keys)

    img = np.empty((362, 174, 224, num), 'f8')  # memory map = correlation map

    if mask == '': # If mask is absent, do computations in the old way
        # create an iterator over one dimensional array to control flow
        for z in range(362):
            t = time.time()
            y_slice = np.memmap(out + '_y_slice.dat', shape=(174, 224, num), dtype='f8', mode='w+')
            slices = np.empty((count, 174, 224), 'f8')
            for index, key in enumerate(keys):
                slices[index] = grp[key][z, :]
                
            if compute.lower() == "multiple":
                Parallel(n_jobs=num_cores)(delayed(calculate_vector)(slices[:,y,:], var, y, y_slice, num) for y in range(174))
            elif compute.lower() == "logistic":
                Parallel(n_jobs=num_cores)(delayed(calculateLogisticVector)(slices[:,y,:], var, y, labels, y_slice, num) for y in range(174))
            
            img[z] = y_slice
            for n, filename in enumerate(out_filenames):
                try:
                    write_image(img[:,:,:,n], filename)
                except:
                    continue
            print(z, round(time.time() - t, 2))   
            
            
    else: # If mask is available, compute only in non-zero mask coordinates
        non_zero_z = np.sort(np.unique(coord[:,0])) # Only do computations in non-zero slices
        for z in non_zero_z:
            t = time.time()
            ind_coord = np.where(coord[:,0] == z)[0] # Rows of coordinate matrix where slice "z" has coordinates
            coord_z = coord[ind_coord,1:] # Array with slice coordinates where to do computations
            y_slice = np.memmap(out + '_y_slice.dat', shape=(174, 224, num), dtype='f8', mode='w+') # Memory map where to store result (number of non-zero mask coordinates,1)
            read_rows = np.unique(coord_z[:,0]) # Rows where to do calculations 
            slices = np.empty((count, 174, 224), 'f8')
            for index, key in enumerate(keys):
                slices[index] = grp[key][z, :]
                
            if compute.lower() == "multiple":
                Parallel(n_jobs=num_cores)(delayed(calculateVectorMask)(slices[:,y,:], var, y, coord_z, y_slice, num) for y in read_rows)
            elif compute.lower() == "logistic":
                Parallel(n_jobs=num_cores)(delayed(calculateLogisticVectorMask)(slices[:,y,:], var, y, labels, coord_z, y_slice, num) for y in read_rows)
            img[z] = y_slice
            for n, filename in enumerate(out_filenames):
                try:
                    write_image(img[:,:,:,n], filename)
                except:
                    continue
            print(z, round(time.time() - t, 2))
        

    print("Total time:", round(time.time() - init, 2), 'seconds')  # Print total time calculating correlation map.
    print("File Name:", out)
    print("Samples", count)
    
    #img[img == 0] = np.nan # Set background values with zeros to NaN (facilitate interpretation)
    for n, filename in enumerate(out_filenames):
        try:
            write_image(img[:,:,:,n], filename)
        except:
            continue
    #write_image(img, out)  # Write correlation map to file
    os.remove(out + '_y_slice.dat')
    f.close()


def calculate_vector(slices, var, y, output, num):
    """
    Function that calculates a Z-Y vector
    @param slices: Numpy array of shape (N, 174, 224) where N is the number of samples.
    @param var: list of values that we will use for the computation
    @param y: Y-index
    @param output: Memory Map
    @param num: number of parameters to compute in the multiple regression
    @return: list Correlated vector corresponding to the Z-Y vector.
    """
    img = np.empty((224, num), 'f8')  # vector which will contain the correlated values
    
    # Smart iterator over slice coordinates with non-zero mask
    it = np.nditer(np.arange(img.shape[0]), flags=['f_index'], op_flags=['readonly'])  # create an iterator over the vector
        
    while not it.finished:
        
        # Set the coefficient value to the final image
        X = sm.add_constant(np.array(var).T)
        model = sm.OLS(slices[:,it.index], X).fit()
        img[it.index,:num//2] = np.array(model.params).astype(float)
        img[it.index,num//2:(-1)] = np.array(model.pvalues).astype(float)
        img[it.index,-1] = model.rsquared_adj
        it.iternext()
                
    output[y,:] = img

def calculateLogisticVector(slices, var, y, labels, output, num):
    """
    Function that calculates a Z-Y vector, applying logistic regression
    @param slices: Numpy array of shape (N, 174, 224) where N is the number of samples.
    @param var: list of values that we will use for the computation
    @param y: Y-index
    @param labels: np.ndarray with labels indicating diabetes status
    @param output: Memory Map
    @param num: number of parameters to compute in the multiple regression
    @return: list Correlated vector corresponding to the Z-Y vector.
    """
    
    img = np.empty((224, num), 'f8')  # vector which will contain the correlated values
    
    # Smart iterator over slice coordinates with non-zero mask
    it = np.nditer(np.arange(img.shape[0]), flags=['f_index'], op_flags=['readonly'])  # create an iterator over the vector
    
    X = np.array(var)
    
    while not it.finished:

        X = np.vstack((slices[:, it.index],X)).T
        X = sm.add_constant(X)
        try:
            model = sm.Logit(labels, X).fit(disp = False)
            img[it.index,:num//2] = np.array(model.params).astype(float)
            img[it.index,num//2:] = np.array(model.pvalues).astype(float)
            img[it.index,num//2:(-1)] = np.array(model.pvalues).astype(float)
            img[it.index,-1] = model.prsquared
        except:	# Singular matrix case,	avoid the code to go to	crap
       	    img[it.index,:num//2] = np.zeros(num//2)
       	    img[it.index,num//2:(-1)] =	np.ones(num//2)
       	    img[it.index,-1] = 0
        it.iternext()
     
    output[y,:] = img
            

def calculateVectorMask(slices, var, y, coord_x, output, num):
    """
    Computes quantity desired over slice, just in coordinates where the mask is non-zero
    
    Parameters
    ----------
    slices: np.ndarray (subjects,coordinates)
        Piece of image volumes along subjects for X and Y coordinates where mask is non-zero    
    var: np.ndarray/list of np.ndarray
        Non-imaging variable(s) against with which to compute the maps
    y: int
        Y voxel where to compute the map
    x: int
        X voxel where to compute the map
    coord_x: np.ndarray 
        X coordinates where to iterate
    output: np.memmap
        Memory-map of slice where result is allocated (originally empty)
    num: int
        Number of parameters to compute in the multiple regression
   
    Returns:
    -------
    output: np.memmap
        Memory-map of slice where result is allocated
    """
    pos = np.where(coord_x[:,0] == y)[0]
    coord_x = coord_x[pos,1] # X indices where to do computations
    img = np.empty((len(coord_x), num), 'f8')  # vector which will contain the correlated values
    
    it = np.nditer(coord_x, flags = ['f_index'], op_flags = ['readonly'])

    while not it.finished:
        
        # Set the coefficient value to the final image
        
        X = np.array(var).T
        X = sm.add_constant(X)
        model = sm.OLS(slices[:,it[0]], X).fit()
        img[it.index,:num//2] = np.array(model.params).astype(float)
        img[it.index,num//2:(-1)] = np.array(model.pvalues).astype(float)
        img[it.index,-1] = model.rsquared_adj
        it.iternext()

       
    output[y, coord_x, :] = img 
    
    
def calculateLogisticVectorMask(slices, var, y, labels, coord_x, output, num):
    """
    Computes quantity desired over slice, just in coordinates where the mask is non-zero,
    applying logistic regression
    
    Parameters
    ----------
    slices: np.ndarray (subjects,coordinates)
        Piece of image volumes along subjects for X and Y coordinates where mask is non-zero    
    var: np.ndarray/list of np.ndarray
        Non-imaging variable(s) against with which to compute the maps
    y: int
        Y voxel where to compute the map
    labels: np.ndarray
        np.ndarray with labels indicating diabetes status
    coord_x: np.ndarray 
        X coordinates where to iterate
    output: np.memmap
        Memory-map of slice where result is allocated (originally empty)
    num: int
        Number of parameters to compute in the multiple regression
   
    Returns:
    -------
    output: np.memmap
        Memory-map of slice where result is allocated
    """
    pos = np.where(coord_x[:,0] == y)[0]
    coord_x = coord_x[pos,1] # X indices where to do computations
    img = np.empty((len(coord_x), num), 'f8')  # vector which will contain the correlated values
    
    it = np.nditer(coord_x, flags = ['f_index'], op_flags = ['readonly'])
    
    while not it.finished:
        X = np.array(var)
        X = np.vstack((slices[:, it[0]],X)).T
        
        try:
            model = sm.Logit(labels, sm.add_constant(X)).fit(disp = False)
            img[it.index,:num//2] = np.array(model.params).astype(float)
            img[it.index,num//2:(-1)] = np.array(model.pvalues).astype(float)
            img[it.index,-1] = model.prsquared
        except: # Singular matrix case, avoid the code to go to crap
            img[it.index,:num//2] = np.zeros(num//2)
            img[it.index,num//2:(-1)] = np.ones(num//2)
            img[it.index,-1] = 0
        it.iternext()
     
    output[y, coord_x, :] = img
    
    
def maskLoader(mask_path):
    """
    Loads a mask if a mask pathfile is given, to reduce computational time
    Saves in an array the coordinates where the mask is non-zero
    
    Parameters
    ----------
    mask_path : str
        filepath of mask image.

    Returns
    -------
    coord: np.ndarray (N,3)
        array with coordinates where the mask is non-zero (where to do computations)

    """
    mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask)
    coord = np.array(np.where(mask > 0)).T
    
    return coord
    

def write_image(img, f_path):
    """
    Function that writes an image from array to file
    @param img: Array that represents the image
    @param f_path: Path with filename where to store the file.
    @return:
    """
    corr_map = sitk.GetImageFromArray(img)  # create image from numpy array information
    corr_map.SetSpacing((2.232142925262451, 2.2321431636810303, 3.000000238418579))  # set meta spacing
    corr_map.SetOrigin((-250.0, -194.19644165039062, -789.0))  # set meta origin
    corr_map.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # set meta direction
    corr_map = sitk.Cast(corr_map, sitk.sitkFloat32)
    sitk.WriteImage(corr_map, f_path)  # write the image to file path


if __name__ == '__main__':
    main(sys.argv[1:])
